import argparse
import contextlib
import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import layout
from embedding_artifacts import (
    ArtifactCompatibilityError,
    EmbeddingArtifactContract,
    artifact_is_reusable,
    publish_embedding_artifact,
    sidecar_path,
)
from input_fingerprint import manifest_fingerprint

from croma.alignment import build_embedding_source_manifest
from croma.metrics.pairs import load_manifest

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    DataLoader = None

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked
except ModuleNotFoundError:
    timm = None
    resolve_data_config = None
    create_transform = None
    SwiGLUPacked = None

try:
    from transformers import AutoImageProcessor, AutoModel
except ModuleNotFoundError:
    AutoImageProcessor = None
    AutoModel = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from model_registry import ModelSpec, _build_model_registry, _parse_models
from progress_utils import progress_bar, progress_write, resolve_progress_mode


# Bump whenever preprocessing, model invocation, or feature reduction changes in a way
# that can alter embeddings without changing the ModelSpec fields below.
_EXTRACTION_CONTRACT_VERSION = 1

_RUDOLFV2_INPUT_SIZE = 224
_RUDOLFV2_NUM_REGISTERS = 8
_RUDOLFV2_NUM_PATCHES = 784
_RUDOLFV2_MEAN = (0.7072, 0.5787, 0.7036)
_RUDOLFV2_STD = (0.2119, 0.2301, 0.1775)

_WAIV_POOLING_CONTRACTS = ("canonical", "cls-mean-patch", "cls-only")
_RUDOLFV2_POOLING_CONTRACTS = ("canonical", "cls-only")
_WAIV_PATCH_TOKENS_BY_MODEL_ID = {
    "wearewaiv/mascaret": 256,
    "wearewaiv/phaet": 196,
}


@dataclasses.dataclass(frozen=True)
class _WaivExtractionConfig:
    input_size: int = 224
    input_dtype: str = "float32"
    scale_uint8: bool = True
    patch_tokens: int = 256

    def details(self, pooling: str = "canonical") -> dict:
        details = {
            "preprocessing": {
                "resize": self.input_size,
                "center_crop": self.input_size,
                "input_dtype": self.input_dtype,
                "input_scaling": (
                    "uint8-to-unit-float" if self.scale_uint8 else "unscaled"
                ),
                "normalization": "checkpoint-config:pixel_mean,pixel_std",
            },
        }
        if pooling == "canonical":
            details["pooling"] = {
                "method": "checkpoint-native:model.encode",
                "output_normalization": "checkpoint-native",
            }
        elif pooling == "cls-mean-patch":
            details["pooling"] = {
                "representation_id": pooling,
                "method": "concatenate-raw-cls-and-mean-patches",
                "patch_tokens": self.patch_tokens,
                "output_normalization": "none",
            }
        elif pooling == "cls-only":
            details["pooling"] = {
                "representation_id": pooling,
                "method": "raw-cls",
                "output_normalization": "none",
            }
        else:
            raise ValueError(
                f"unknown Waiv pooling contract {pooling!r}; "
                f"expected one of {list(_WAIV_POOLING_CONTRACTS)}"
            )
        return details

    def build_transform(self, v2, model):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.input_size),
                v2.CenterCrop(self.input_size),
                v2.ToDtype(getattr(torch, self.input_dtype), scale=self.scale_uint8),
                v2.Normalize(
                    mean=model.config.pixel_mean,
                    std=model.config.pixel_std,
                ),
            ]
        )

    def embed(self, model, batch, pooling: str = "canonical"):
        if pooling == "canonical":
            return model.encode(batch)
        output = model(pixel_values=batch)
        tokens = output.last_hidden_state
        expected_tokens = 1 + self.patch_tokens
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Waiv final-layer forward must return "
                f"{expected_tokens} tokens (CLS + {self.patch_tokens} patches); "
                f"got shape {tuple(tokens.shape)}."
            )
        if pooling == "cls-mean-patch":
            return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
        if pooling == "cls-only":
            return tokens[:, 0]
        raise ValueError(
            f"unknown Waiv pooling contract {pooling!r}; "
            f"expected one of {list(_WAIV_POOLING_CONTRACTS)}"
        )


@dataclasses.dataclass(frozen=True)
class _RudolfV2ExtractionConfig:
    input_size: int = _RUDOLFV2_INPUT_SIZE
    interpolation: str = "bicubic"
    antialias: bool = True
    input_dtype: str = "float32"
    scale_uint8: bool = True
    normalization_mean: tuple[float, float, float] = _RUDOLFV2_MEAN
    normalization_std: tuple[float, float, float] = _RUDOLFV2_STD
    register_tokens: int = _RUDOLFV2_NUM_REGISTERS
    patch_tokens: int = _RUDOLFV2_NUM_PATCHES

    def details(self, pooling: str = "canonical") -> dict:
        size = [self.input_size, self.input_size]
        details = {
            "preprocessing": {
                "resize": size,
                "resize_interpolation": self.interpolation,
                "resize_antialias": self.antialias,
                "center_crop": list(size),
                "input_dtype": self.input_dtype,
                "input_scaling": (
                    "uint8-to-unit-float" if self.scale_uint8 else "unscaled"
                ),
                "normalization_mean": list(self.normalization_mean),
                "normalization_std": list(self.normalization_std),
            },
        }
        if pooling == "canonical":
            details["pooling"] = {
                "method": "concatenate-cls-and-mean-patches",
                "register_tokens_excluded": self.register_tokens,
                "patch_tokens": self.patch_tokens,
            }
        elif pooling == "cls-only":
            details["pooling"] = {
                "representation_id": pooling,
                "method": "raw-cls",
                "register_tokens_excluded": self.register_tokens,
                "patch_tokens": self.patch_tokens,
                "output_normalization": "none",
            }
        else:
            raise ValueError(
                f"unknown RudolfV 2 pooling contract {pooling!r}; "
                f"expected one of {list(_RUDOLFV2_POOLING_CONTRACTS)}"
            )
        return details

    def build_transform(self, v2):
        size = (self.input_size, self.input_size)
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    size,
                    interpolation=getattr(
                        v2.InterpolationMode,
                        self.interpolation.upper(),
                    ),
                    antialias=self.antialias,
                ),
                v2.CenterCrop(size),
                v2.ToDtype(getattr(torch, self.input_dtype), scale=self.scale_uint8),
                v2.Normalize(
                    mean=self.normalization_mean,
                    std=self.normalization_std,
                ),
            ]
        )

    def embed(self, model, batch, pooling: str = "canonical"):
        tokens = model.model.encode(batch)["last_hidden_state"]
        prefix_tokens = 1 + self.register_tokens
        expected_tokens = prefix_tokens + self.patch_tokens
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"RudolfV 2 published native forward must return {expected_tokens} "
                f"tokens (CLS + {self.register_tokens} registers + "
                f"{self.patch_tokens} patches); "
                f"got shape {tuple(tokens.shape)}."
            )
        if pooling == "canonical":
            patch_tokens = tokens[:, prefix_tokens:]
            return torch.cat([tokens[:, 0], patch_tokens.mean(1)], dim=-1)
        if pooling == "cls-only":
            return tokens[:, 0]
        raise ValueError(
            f"unknown RudolfV 2 pooling contract {pooling!r}; "
            f"expected one of {list(_RUDOLFV2_POOLING_CONTRACTS)}"
        )


_BACKEND_EXTRACTION_CONFIGS = {
    "waiv": _WaivExtractionConfig(),
    "rudolfv2": _RudolfV2ExtractionConfig(),
}


def _waiv_extraction_config(spec: ModelSpec) -> _WaivExtractionConfig:
    try:
        patch_tokens = _WAIV_PATCH_TOKENS_BY_MODEL_ID[spec.model_id]
    except KeyError:
        raise ValueError(f"unknown Waiv token layout for model {spec.model_id!r}") from None
    return dataclasses.replace(
        _BACKEND_EXTRACTION_CONFIGS["waiv"],
        patch_tokens=int(patch_tokens),
    )


def _extract_timm_features(out, extract: str):
    if extract == "cls":
        return out[:, 0] if out.ndim == 3 else out
    if extract == "cls_and_patch":
        if out.ndim != 3:
            raise RuntimeError("cls_and_patch extraction expects 3D token output.")
        return torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
    if extract == "virchow":
        if out.ndim != 3:
            raise RuntimeError("virchow extraction expects 3D token output.")
        return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)
    raise ValueError(f"Unknown extract mode: {extract}")


_GPFM_ARCH = "vit_large_patch14_dinov2.lvd142m"
_GPFM_CHECKPOINT = "GPFM.pth"
_GPFM_WRAPPER_KEYS = ("model", "teacher", "student", "state_dict", "teacher_backbone")
_GPFM_PREFIXES = ("module.", "backbone.")


def _unwrap_gpfm_state_dict(payload):
    """Reduce a GPFM checkpoint payload to a bare state_dict (mirrors slide2vec)."""
    state = payload
    for wrapper in _GPFM_WRAPPER_KEYS:
        if (
            isinstance(state, Mapping)
            and wrapper in state
            and isinstance(state[wrapper], Mapping)
        ):
            state = state[wrapper]
            break
    if not isinstance(state, Mapping):
        raise ValueError(
            "Unexpected GPFM checkpoint payload: expected a state_dict mapping, "
            f"got {type(state)}"
        )
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in _GPFM_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns:
        raise ValueError("Manifest must contain an 'image_path' column.")
    return df.copy()


def _resolve_image_paths(manifest_path: Path, image_path_map: Path | None) -> list[str] | None:
    """Resolve optional access-only paths without changing manifest identity."""

    if image_path_map is None:
        return None
    manifest = _load_manifest(manifest_path)
    mapping = pd.read_csv(image_path_map)
    required = ("sample_id", "canonical_image_path", "access_path")
    missing = [column for column in required if column not in mapping.columns]
    if missing:
        raise ValueError(f"image path map is missing required columns: {missing}")

    canonical_keys = list(
        zip(
            manifest["sample_id"].astype(str),
            manifest["image_path"].astype(str),
            strict=True,
        )
    )
    mapped_keys = list(
        zip(
            mapping["sample_id"].astype(str),
            mapping["canonical_image_path"].astype(str),
            strict=True,
        )
    )
    if len(set(mapped_keys)) != len(mapped_keys):
        raise ValueError("image path map contains duplicate canonical tile identities")
    access_paths = mapping["access_path"].astype(str).tolist()
    if len(set(access_paths)) != len(access_paths):
        raise ValueError("image path map must map canonical tiles one-to-one to access paths")
    if set(mapped_keys) != set(canonical_keys):
        missing_keys = len(set(canonical_keys) - set(mapped_keys))
        extra_keys = len(set(mapped_keys) - set(canonical_keys))
        raise ValueError(
            "image path map must cover the canonical manifest exactly: "
            f"missing={missing_keys}, extra={extra_keys}"
        )

    by_key = dict(zip(mapped_keys, access_paths, strict=True))
    resolved = [by_key[key] for key in canonical_keys]
    missing_files = [path for path in resolved if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"image path map resolves to {len(missing_files)} missing access path(s); "
            f"first: {missing_files[0]}"
        )
    return resolved


def _model_extraction_details(spec: ModelSpec, pooling: str = "canonical") -> dict:
    config = _BACKEND_EXTRACTION_CONFIGS.get(spec.backend)
    if config is None:
        if pooling != "canonical":
            raise ValueError(
                f"explicit pooling is unsupported for backend {spec.backend!r}"
            )
        return {}
    if spec.backend == "waiv":
        return _waiv_extraction_config(spec).details(pooling)
    if spec.backend == "rudolfv2":
        return config.details(pooling)
    if pooling != "canonical":
        raise ValueError(
            f"explicit pooling is unsupported for backend {spec.backend!r}"
        )
    return config.details()


def build_embedding_artifact_contract(
    *,
    manifest_path: Path,
    spec: ModelSpec,
    batch_size: int,
    device_arg: str,
    pooling: str = "canonical",
) -> EmbeddingArtifactContract:
    """Build the exact artifact contract expected for one extraction invocation."""

    manifest = _load_manifest(manifest_path)
    manifest_fp = manifest_fingerprint(manifest)
    revision = spec.checkpoint_revision
    if revision is not None and (
        len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise ValueError(
            f"checkpoint revision must be an immutable 40-character commit SHA, got {revision!r}"
        )
    uses_cuda = (
        bool(torch is not None and torch.cuda.is_available())
        if device_arg == "auto"
        else str(device_arg).startswith("cuda")
    )
    return EmbeddingArtifactContract(
        checkpoint_revision=revision,
        extraction_contract={
            "version": _EXTRACTION_CONTRACT_VERSION,
            "backend": str(spec.backend),
            "model_id": str(spec.model_id),
            "extract": str(spec.extract),
            "timm_kwargs": dict(spec.timm_kwargs),
            **_model_extraction_details(spec, pooling),
        },
        precision=(
            "mixed-float16"
            if spec.mixed_precision and uses_cuda
            else "float32"
        ),
        manifest_fingerprint=manifest_fp,
        batch_size=int(batch_size),
        output_dtype="float32",
        output_shape=(
            int(len(manifest)),
            (
                int(spec.embedding_dim) * 2
                if spec.backend == "waiv"
                and pooling == "cls-mean-patch"
                and spec.embedding_dim is not None
                else int(spec.embedding_dim) // 2
                if spec.backend == "rudolfv2"
                and pooling == "cls-only"
                and spec.embedding_dim is not None
                else int(spec.embedding_dim) if spec.embedding_dim is not None else None
            ),
        ),
    )


def _device_from_arg(device_arg: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for embedding extraction")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class TileDataset:
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def _load_model_and_transform(spec: ModelSpec, device, *, pooling: str = "canonical"):
    if spec.backend == "timm":
        timm_kwargs = dict(spec.timm_kwargs)
        model_id = spec.model_id
        if spec.checkpoint_revision is not None:
            if model_id.startswith(("hf-hub:", "hf_hub:")):
                model_id = f"{model_id}@{spec.checkpoint_revision}"
            else:
                timm_kwargs["pretrained_cfg_overlay"] = {
                    "hf_hub_id": f"timm/{model_id}@{spec.checkpoint_revision}"
                }
        if timm_kwargs.get("mlp_layer") == "SwiGLUPacked":
            timm_kwargs["mlp_layer"] = SwiGLUPacked
        if timm_kwargs.get("act_layer") == "SiLU":
            timm_kwargs["act_layer"] = torch.nn.SiLU

        model = timm.create_model(model_id, pretrained=True, **timm_kwargs)
        model.eval().to(device)
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        def embed_fn(batch):
            out = model.forward_features(batch)
            return _extract_timm_features(out, spec.extract)

        return model, transform, embed_fn

    if spec.backend == "waiv":
        from torchvision.transforms import v2

        config = _waiv_extraction_config(spec)
        model = AutoModel.from_pretrained(
            spec.model_id,
            trust_remote_code=True,
            revision=spec.checkpoint_revision,
        ).eval()
        model.to(device)
        transform = config.build_transform(v2, model)

        def embed_fn(batch):
            return config.embed(model, batch, pooling)

        return model, transform, embed_fn

    if spec.backend == "hf_auto":
        revision_kwargs = (
            {"revision": spec.checkpoint_revision}
            if spec.checkpoint_revision is not None
            else {}
        )
        processor = AutoImageProcessor.from_pretrained(
            spec.model_id, trust_remote_code=True, **revision_kwargs
        )
        model = AutoModel.from_pretrained(
            spec.model_id, trust_remote_code=True, **revision_kwargs
        )
        model.eval().to(device)
        transform = lambda img: processor(img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        def embed_fn(batch):
            out = model(pixel_values=batch).last_hidden_state
            if spec.extract == "cls":
                return out[:, 0, :]
            if spec.extract == "cls_and_patch":
                return torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
            raise ValueError(
                f"Unsupported extract mode for hf_auto backend: {spec.extract}"
            )

        return model, transform, embed_fn

    if spec.backend == "rudolfv2":
        from torchvision.transforms import v2

        config = _BACKEND_EXTRACTION_CONFIGS["rudolfv2"]
        revision_kwargs = (
            {"revision": spec.checkpoint_revision}
            if spec.checkpoint_revision is not None
            else {}
        )
        model = AutoModel.from_pretrained(
            spec.model_id, trust_remote_code=True, **revision_kwargs
        )
        model.eval().to(device)
        transform = config.build_transform(v2)

        def embed_fn(batch):
            return config.embed(model, batch, pooling)

        return model, transform, embed_fn

    if spec.backend == "midnight":
        from torchvision.transforms import v2

        revision_kwargs = (
            {"revision": spec.checkpoint_revision}
            if spec.checkpoint_revision is not None
            else {}
        )
        model = AutoModel.from_pretrained(spec.model_id, **revision_kwargs)
        model.eval().to(device)
        transform = v2.Compose(
            [
                v2.Resize(224),
                v2.CenterCrop(224),
                v2.ToTensor(),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        def embed_fn(batch):
            out = model(pixel_values=batch).last_hidden_state
            if spec.extract == "cls_and_patch":
                return torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
            if spec.extract == "cls":
                return out[:, 0, :]
            raise ValueError(
                f"Unsupported extract mode for midnight backend: {spec.extract}"
            )

        return model, transform, embed_fn

    if spec.backend == "conch_v1":
        from conch.open_clip_custom import create_model_from_pretrained

        if spec.checkpoint_revision is None:
            checkpoint_path = "hf_hub:MahmoodLab/conch"
        else:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id=spec.model_id,
                filename="pytorch_model.bin",
                revision=spec.checkpoint_revision,
            )
        model, transform = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path
        )
        model.eval().to(device)

        def embed_fn(batch):
            return model.encode_image(batch, proj_contrast=False, normalize=False)

        return model, transform, embed_fn

    if spec.backend == "conch_v1_5":
        from trident.patch_encoder_models import encoder_factory

        if spec.checkpoint_revision is None:
            encoder = encoder_factory(model_name="conch_v15")
        else:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id=spec.model_id,
                filename="pytorch_model_vision.bin",
                revision=spec.checkpoint_revision,
            )
            encoder = encoder_factory(
                model_name="conch_v15", weights_path=checkpoint_path
            )
        encoder.eval().to(device)

        def embed_fn(batch):
            return encoder(batch)

        return encoder, encoder.eval_transforms, embed_fn

    if spec.backend == "gpfm":
        from huggingface_hub import hf_hub_download

        model = timm.create_model(
            _GPFM_ARCH,
            pretrained=False,
            img_size=224,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        revision_kwargs = (
            {"revision": spec.checkpoint_revision}
            if spec.checkpoint_revision is not None
            else {}
        )
        checkpoint_path = hf_hub_download(
            repo_id=spec.model_id,
            filename=_GPFM_CHECKPOINT,
            **revision_kwargs,
        )
        # weights_only=False: GPFM.pth is a pickled checkpoint (executes code on load);
        # safe only from the trusted majiabo/GPFM repo. Do not repoint at untrusted repos.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Invalid GPFM checkpoint payload: expected a dict, got {type(payload)}"
            )
        model.load_state_dict(_unwrap_gpfm_state_dict(payload), strict=True)
        model.eval().to(device)
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        def embed_fn(batch):
            out = model.forward_features(batch)
            return _extract_timm_features(out, spec.extract)

        return model, transform, embed_fn

    if spec.backend == "musk":
        from musk import modeling, utils  # noqa: F401 - modeling registers the timm arch
        from timm.models import create_model
        from torchvision.transforms import v2

        model = create_model("musk_large_patch16_384").eval()
        if spec.checkpoint_revision is None:
            checkpoint_path = spec.model_id
        else:
            from huggingface_hub import hf_hub_download

            repo_id = spec.model_id.removeprefix("hf_hub:")
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                revision=spec.checkpoint_revision,
            )
        utils.load_model_and_may_interpolate(
            checkpoint_path, model, "model|module", ""
        )
        model.to(device)
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    384, interpolation=v2.InterpolationMode.BICUBIC, antialias=True
                ),
                v2.CenterCrop(384),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        ms_aug = spec.extract == "ms_aug"

        def embed_fn(batch):
            return model(
                image=batch, with_head=False, out_norm=False, ms_aug=ms_aug
            )[0]

        return model, transform, embed_fn

    if spec.backend == "genbio":
        from torchvision.transforms import v2

        revision_kwargs = (
            {"revision": spec.checkpoint_revision}
            if spec.checkpoint_revision is not None
            else {}
        )
        model = AutoModel.from_pretrained(
            spec.model_id, trust_remote_code=True, **revision_kwargs
        )
        model.eval().to(device)
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    (224, 224),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187)),
            ]
        )

        def embed_fn(batch):
            # GenBio's forward returns [B, 4608] embeddings directly (not last_hidden_state).
            return model(batch)

        return model, transform, embed_fn

    raise ValueError(f"Unknown backend: {spec.backend}")


def _output_path_in_dir(manifest: Path, output_dir: Path, model_name: str) -> Path:
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return output_dir / f"{safe_model}.npy"


def embed_manifest(
    manifest_path: Path,
    output_path: Path,
    spec: ModelSpec,
    batch_size: int,
    num_workers: int,
    device_arg: str,
    artifact_contract: EmbeddingArtifactContract | None = None,
    progress_enabled: bool | None = None,
    tile_progress_leave: bool = True,
    image_paths: list[str] | None = None,
    pooling: str = "canonical",
) -> tuple[Path, tuple[int, int]]:
    manifest = _load_manifest(manifest_path)
    if artifact_contract is None:
        artifact_contract = build_embedding_artifact_contract(
            manifest_path=manifest_path,
            spec=spec,
            batch_size=batch_size,
            device_arg=device_arg,
            pooling=pooling,
        )
    device = _device_from_arg(device_arg)
    progress_on = bool(progress_enabled) if progress_enabled is not None else True
    resolved_image_paths = manifest["image_path"].astype(str).tolist()
    if image_paths is not None:
        if len(image_paths) != len(manifest):
            raise ValueError(
                "image access paths must have one entry per canonical manifest row: "
                f"expected {len(manifest)}, got {len(image_paths)}"
            )
        resolved_image_paths = [str(path) for path in image_paths]

    progress_write(f"[embed] manifest: {manifest_path}", enabled=progress_on)
    progress_write(f"[embed] samples: {len(manifest)}", enabled=progress_on)
    progress_write(
        f"[embed] backend/model: {spec.backend} / {spec.model_id}", enabled=progress_on
    )
    progress_write(f"[embed] device: {device}", enabled=progress_on)

    _model, transform, embed_fn = _load_model_and_transform(
        spec, device, pooling=pooling
    )
    dataset = TileDataset(resolved_image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(str(device).startswith("cuda")),
    )

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if spec.mixed_precision and str(device).startswith("cuda")
        else contextlib.nullcontext()
    )

    all_emb = []
    total = len(dataset)
    with torch.inference_mode(), autocast:
        with progress_bar(
            total=total,
            desc="[embed] tiles",
            unit="img",
            enabled=progress_on,
            leave=tile_progress_leave,
        ) as pbar:
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                emb = embed_fn(batch).float()
                all_emb.append(emb.cpu().numpy())
                pbar.update(len(batch))
    arr = np.concatenate(all_emb, axis=0)
    if int(arr.shape[0]) != artifact_contract.output_shape[0] or (
        artifact_contract.output_shape[1] is not None
        and int(arr.shape[1]) != artifact_contract.output_shape[1]
    ):
        raise ValueError(
            "extracted embedding shape does not match the expected artifact shape: "
            f"expected {artifact_contract.output_shape}, got {tuple(arr.shape)}"
        )
    published_contract = dataclasses.replace(
        artifact_contract,
        output_shape=(int(arr.shape[0]), int(arr.shape[1])),
    )
    publish_embedding_artifact(output_path, arr, published_contract)
    sidecar = sidecar_path(output_path)
    progress_write(
        f"[embed] saved embeddings: {output_path} shape={arr.shape}",
        enabled=progress_on,
    )
    progress_write(f"[embed] saved metadata  : {sidecar}", enabled=progress_on)
    return output_path, (int(arr.shape[0]), int(arr.shape[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract tile embeddings for a tileset. This is the sole writer of "
            "output/embeddings/<tileset>/: it owns manifest.csv (the row-order contract) "
            "and every <Model>.npy aligned to it."
        )
    )
    parser.add_argument(
        "--tileset",
        required=True,
        help="Tileset name; embeddings land in output/embeddings/<tileset>/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Source manifest CSV. Required only the first time a tileset is embedded, "
            "to derive its manifest.csv; ignored once that exists."
        ),
    )
    parser.add_argument(
        "--confounder-column",
        default="medical_center",
        help="Source-manifest column holding the confounder (used to derive manifest.csv).",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model names (e.g. Virchow2,UNI,Phikon-v2).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--image-path-map",
        type=Path,
        default=None,
        help=(
            "Optional CSV with sample_id, canonical_image_path, and access_path. "
            "It may redirect tile reads to a one-to-one local mirror without changing "
            "the frozen manifest or artifact provenance."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output file."
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress display mode: auto=TTY only, on=always, off=never.",
    )
    return parser.parse_args()


def _resolve_specs(model_names: list[str]) -> list[tuple[str, ModelSpec]]:
    model_registry = _build_model_registry()
    unknown = [m for m in model_names if m not in model_registry]
    if unknown:
        available = ", ".join(sorted(model_registry))
        raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")
    return [(m, model_registry[m]) for m in model_names]


def _resolve_tileset_manifest(
    tileset: str, source_manifest: Path | None, confounder_column: str, progress: bool
) -> Path:
    """Return the tileset's manifest.csv, deriving it on first use.

    The manifest is written once and then frozen: it is the row-order contract every
    ``<Model>.npy`` in the directory is aligned to, so re-deriving it under a different
    source would silently invalidate the matrices already on disk.
    """
    manifest_path = layout.tileset_manifest(tileset)
    if manifest_path.exists():
        return manifest_path
    if source_manifest is None:
        raise SystemExit(
            f"tileset '{tileset}' has no {manifest_path}; pass --manifest <source csv> "
            "the first time you embed it"
        )
    frame = load_manifest(str(source_manifest), confounder_column=confounder_column)
    tileset_manifest, _ = build_embedding_source_manifest(frame)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tileset_manifest.to_csv(manifest_path, index=False)
    progress_write(
        f"[embed] derived {manifest_path} ({len(tileset_manifest)} rows) "
        f"from {source_manifest}",
        enabled=progress,
    )
    return manifest_path


def main():
    args = parse_args()
    progress_enabled = resolve_progress_mode(str(args.progress))
    model_names = _parse_models(args.models)
    model_specs = _resolve_specs(model_names)

    output_dir = layout.embeddings_dir(str(args.tileset))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _resolve_tileset_manifest(
        str(args.tileset), args.manifest, str(args.confounder_column), progress_enabled
    )
    image_paths: list[str] | None = None

    progress_write(
        f"[embed] models: {', '.join(model_names)}", enabled=progress_enabled
    )
    progress_write(f"[embed] tileset: {args.tileset}", enabled=progress_enabled)
    progress_write(f"[embed] output_dir: {output_dir}", enabled=progress_enabled)

    statuses: list[dict] = []
    with progress_bar(
        total=len(model_specs),
        desc="[embed] models",
        unit="model",
        enabled=progress_enabled,
    ) as model_bar:
        for model_name, spec in model_specs:
            model_bar.set_postfix_str(f"{model_name}:prepare")
            progress_write(
                f"\n[embed] === model: {model_name} ===", enabled=progress_enabled
            )
            output = _output_path_in_dir(manifest_path, output_dir, model_name)
            try:
                artifact_contract = build_embedding_artifact_contract(
                    manifest_path=manifest_path,
                    spec=spec,
                    batch_size=int(args.batch_size),
                    device_arg=str(args.device),
                )
                if not args.force and artifact_is_reusable(output, artifact_contract):
                    progress_write(
                        f"[embed] compatible artifact exists, skipping: {output}",
                        enabled=progress_enabled,
                    )
                    statuses.append(
                        {
                            "model": model_name,
                            "status": "skipped",
                            "output": str(output),
                        }
                    )
                    continue

                if args.image_path_map is not None and image_paths is None:
                    image_paths = _resolve_image_paths(manifest_path, args.image_path_map)
                model_bar.set_postfix_str(f"{model_name}:extract")
                embed_manifest(
                    manifest_path=manifest_path,
                    output_path=output,
                    spec=spec,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    device_arg=str(args.device),
                    artifact_contract=artifact_contract,
                    progress_enabled=progress_enabled,
                    tile_progress_leave=False,
                    image_paths=image_paths,
                )
                statuses.append(
                    {"model": model_name, "status": "ok", "output": str(output)}
                )
            except Exception as exc:  # noqa: BLE001
                progress_write(
                    f"[embed] failed for model '{model_name}': {exc}",
                    enabled=progress_enabled,
                )
                statuses.append(
                    {
                        "model": model_name,
                        "status": "failed",
                        "output": str(output),
                        "error": str(exc),
                    }
                )
            finally:
                model_bar.update(1)

    n_ok = sum(1 for s in statuses if s["status"] == "ok")
    n_skip = sum(1 for s in statuses if s["status"] == "skipped")
    n_fail = sum(1 for s in statuses if s["status"] == "failed")
    progress_write("\n[embed] === summary ===", enabled=progress_enabled)
    progress_write(
        f"[embed] ok={n_ok} skipped={n_skip} failed={n_fail}", enabled=progress_enabled
    )
    for s in statuses:
        error_suffix = f" | error={s['error']}" if s["status"] == "failed" else ""
        progress_write(
            f"[embed] {s['model']}: {s['status']} -> {s['output']}{error_suffix}",
            enabled=progress_enabled,
        )

    if n_fail > 0:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
