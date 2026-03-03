import argparse
import contextlib
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

from progress_utils import progress_bar, progress_write, resolve_progress_mode


@dataclasses.dataclass
class ModelSpec:
    backend: str  # "timm", "hf_auto", "conch_v1", "conch_v1_5", or "midnight"
    model_id: str
    extract: str = "cls"
    timm_kwargs: dict = dataclasses.field(default_factory=dict)
    mixed_precision: bool = False


def _build_model_registry():
    virchow_kwargs = {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}
    uni2h_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": "SwiGLUPacked",
        "act_layer": "SiLU",
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    h0_mini_kwargs = {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}

    return {
        "Virchow2": ModelSpec(
            backend="timm",
            model_id="hf-hub:paige-ai/Virchow2",
            extract="virchow",
            timm_kwargs=virchow_kwargs,
            mixed_precision=True,
        ),
        "Virchow": ModelSpec(
            backend="timm",
            model_id="hf-hub:paige-ai/Virchow",
            extract="cls_and_patch",
            timm_kwargs=virchow_kwargs,
            mixed_precision=True,
        ),
        "UNI2-h": ModelSpec(
            backend="timm",
            model_id="hf-hub:MahmoodLab/UNI2-h",
            extract="cls",
            timm_kwargs=uni2h_kwargs,
        ),
        "UNI": ModelSpec(
            backend="timm",
            model_id="hf-hub:MahmoodLab/uni",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
        ),
        "CONCHv1.5": ModelSpec(
            backend="conch_v1_5",
            model_id="MahmoodLab/conch-v1.5",
            extract="raw",
            mixed_precision=True,
        ),
        "CONCH": ModelSpec(
            backend="conch_v1",
            model_id="MahmoodLab/conch",
            extract="raw",
        ),
        "H-optimus-1": ModelSpec(
            backend="timm",
            model_id="hf-hub:bioptimus/H-optimus-1",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
            mixed_precision=True,
        ),
        "H-optimus-0": ModelSpec(
            backend="timm",
            model_id="hf-hub:bioptimus/H-optimus-0",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
            mixed_precision=True,
        ),
        "H0-mini": ModelSpec(
            backend="timm",
            model_id="hf-hub:bioptimus/H0-mini",
            extract="virchow",
            timm_kwargs=h0_mini_kwargs,
            mixed_precision=True,
        ),
        "Prov-GigaPath": ModelSpec(
            backend="timm",
            model_id="hf-hub:prov-gigapath/prov-gigapath",
            extract="cls",
        ),
        "Midnight-12k": ModelSpec(
            backend="midnight",
            model_id="kaiko-ai/midnight",
            extract="cls_and_patch",
        ),
        "Prost40M": ModelSpec(
            backend="timm",
            model_id="hf-hub:waticlems/Prost40M",
            extract="cls",
            timm_kwargs={"patch_size": 14, "img_size": 224, "num_classes": 0},
        ),
        "Phikon": ModelSpec(
            backend="hf_auto",
            model_id="owkin/phikon",
            extract="cls",
        ),
        "Phikon-v2": ModelSpec(
            backend="hf_auto",
            model_id="owkin/phikon-v2",
            extract="cls",
        ),
        "Hibou-L": ModelSpec(
            backend="hf_auto",
            model_id="histai/hibou-L",
            extract="cls",
        ),
        "Hibou-B": ModelSpec(
            backend="hf_auto",
            model_id="histai/hibou-b",
            extract="cls",
        ),
    }


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


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns:
        raise ValueError("Manifest must contain an 'image_path' column.")
    return df.copy()


def _device_from_arg(device_arg: str):
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


def _load_model_and_transform(spec: ModelSpec, device):
    if spec.backend == "timm":
        timm_kwargs = dict(spec.timm_kwargs)
        if timm_kwargs.get("mlp_layer") == "SwiGLUPacked":
            timm_kwargs["mlp_layer"] = SwiGLUPacked
        if timm_kwargs.get("act_layer") == "SiLU":
            timm_kwargs["act_layer"] = torch.nn.SiLU

        model = timm.create_model(spec.model_id, pretrained=True, **timm_kwargs)
        model.eval().to(device)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        def embed_fn(batch):
            out = model.forward_features(batch)
            return _extract_timm_features(out, spec.extract)

        return model, transform, embed_fn

    if spec.backend == "hf_auto":
        processor = AutoImageProcessor.from_pretrained(spec.model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(spec.model_id, trust_remote_code=True)
        model.eval().to(device)
        transform = lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        def embed_fn(batch):
            out = model(pixel_values=batch).last_hidden_state
            if spec.extract == "cls":
                return out[:, 0, :]
            if spec.extract == "cls_and_patch":
                return torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
            raise ValueError(f"Unsupported extract mode for hf_auto backend: {spec.extract}")

        return model, transform, embed_fn

    if spec.backend == "midnight":
        from torchvision.transforms import v2

        model = AutoModel.from_pretrained(spec.model_id)
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
            raise ValueError(f"Unsupported extract mode for midnight backend: {spec.extract}")

        return model, transform, embed_fn

    if spec.backend == "conch_v1":
        from conch.open_clip_custom import create_model_from_pretrained

        model, transform = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        model.eval().to(device)

        def embed_fn(batch):
            return model.encode_image(batch, proj_contrast=False, normalize=False)

        return model, transform, embed_fn

    if spec.backend == "conch_v1_5":
        from trident.patch_encoder_models import encoder_factory

        encoder = encoder_factory(model_name="conch_v15")
        encoder.eval().to(device)

        def embed_fn(batch):
            return encoder(batch)

        return encoder, encoder.eval_transforms, embed_fn

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
    progress_enabled: bool | None = None,
    tile_progress_leave: bool = True,
) -> tuple[Path, tuple[int, int]]:
    manifest = _load_manifest(manifest_path)
    device = _device_from_arg(device_arg)
    progress_on = bool(progress_enabled) if progress_enabled is not None else True

    progress_write(f"[embed] manifest: {manifest_path}", enabled=progress_on)
    progress_write(f"[embed] samples: {len(manifest)}", enabled=progress_on)
    progress_write(f"[embed] backend/model: {spec.backend} / {spec.model_id}", enabled=progress_on)
    progress_write(f"[embed] device: {device}", enabled=progress_on)

    _model, transform, embed_fn = _load_model_and_transform(spec, device)
    dataset = TileDataset(manifest["image_path"].tolist(), transform)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)

    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "n_samples": int(arr.shape[0]),
                "embedding_dim": int(arr.shape[1]),
                "backend": spec.backend,
                "model_id": spec.model_id,
                "extract": spec.extract,
                "mixed_precision": bool(spec.mixed_precision),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    progress_write(f"[embed] saved embeddings: {output_path} shape={arr.shape}", enabled=progress_on)
    progress_write(f"[embed] saved metadata  : {sidecar}", enabled=progress_on)
    return output_path, (int(arr.shape[0]), int(arr.shape[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tile embeddings from a manifest CSV (image_path column)."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Required output directory for embeddings.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model names (e.g. Virchow2,UNI,Phikon-v2).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file.")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress display mode: auto=TTY only, on=always, off=never.",
    )
    return parser.parse_args()


def _parse_models(raw_models: str) -> list[str]:
    models = [m.strip() for m in str(raw_models).split(",")]
    if any(not m for m in models):
        raise ValueError("Invalid --models value: empty model name detected.")
    if len(set(models)) != len(models):
        raise ValueError("Invalid --models value: duplicate model names detected.")
    return models


def _resolve_specs(model_names: list[str]) -> list[tuple[str, ModelSpec]]:
    model_registry = _build_model_registry()
    unknown = [m for m in model_names if m not in model_registry]
    if unknown:
        available = ", ".join(sorted(model_registry))
        raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")
    return [(m, model_registry[m]) for m in model_names]


def main():
    args = parse_args()
    progress_enabled = resolve_progress_mode(str(args.progress))
    model_names = _parse_models(args.models)
    model_specs = _resolve_specs(model_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_write(f"[embed] models: {', '.join(model_names)}", enabled=progress_enabled)
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
            progress_write(f"\n[embed] === model: {model_name} ===", enabled=progress_enabled)
            output = _output_path_in_dir(args.manifest, output_dir, model_name)
            if output.exists() and not args.force:
                progress_write(f"[embed] output exists, skipping: {output}", enabled=progress_enabled)
                statuses.append({"model": model_name, "status": "skipped", "output": str(output)})
                model_bar.update(1)
                continue

            try:
                model_bar.set_postfix_str(f"{model_name}:extract")
                embed_manifest(
                    manifest_path=args.manifest,
                    output_path=output,
                    spec=spec,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    device_arg=str(args.device),
                    progress_enabled=progress_enabled,
                    tile_progress_leave=False,
                )
                statuses.append({"model": model_name, "status": "ok", "output": str(output)})
            except Exception as exc:  # noqa: BLE001
                progress_write(f"[embed] failed for model '{model_name}': {exc}", enabled=progress_enabled)
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
    progress_write(f"[embed] ok={n_ok} skipped={n_skip} failed={n_fail}", enabled=progress_enabled)
    for s in statuses:
        error_suffix = f" | error={s['error']}" if s["status"] == "failed" else ""
        progress_write(f"[embed] {s['model']}: {s['status']} -> {s['output']}{error_suffix}", enabled=progress_enabled)

    if n_fail > 0:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
