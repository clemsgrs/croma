import contextlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_registry as mr


def test_registry_includes_conch_and_midnight_models() -> None:
    registry = mr._build_model_registry()

    assert "CONCH" in registry
    assert registry["CONCH"].backend == "conch_v1"
    assert registry["CONCH"].extract == "raw"

    assert "CONCHv1.5" in registry
    assert registry["CONCHv1.5"].backend == "conch_v1_5"
    assert registry["CONCHv1.5"].extract == "raw"
    assert registry["CONCHv1.5"].mixed_precision is True

    assert "Midnight-12k" in registry
    assert registry["Midnight-12k"].backend == "midnight"
    assert registry["Midnight-12k"].model_id == "kaiko-ai/midnight"
    assert registry["Midnight-12k"].extract == "cls_and_patch"


def test_registry_includes_ported_slide2vec_models() -> None:
    registry = mr._build_model_registry()

    assert registry["mSTAR"].backend == "timm"
    assert registry["mSTAR"].model_id == "hf-hub:Wangyh/mSTAR"
    assert registry["mSTAR"].extract == "cls"

    assert registry["DINOv2-B"].backend == "timm"
    assert registry["DINOv2-B"].model_id == "vit_base_patch14_dinov2.lvd142m"
    assert registry["DINOv2-B"].extract == "cls"

    assert registry["GPFM"].backend == "gpfm"
    assert registry["GPFM"].model_id == "majiabo/GPFM"

    assert registry["MUSK"].backend == "musk"
    assert registry["MUSK"].model_id == "hf_hub:xiangjx/musk"
    assert registry["MUSK"].extract == "ms_aug"
    assert registry["MUSK"].mixed_precision is True

    assert registry["GenBio-PathFM"].backend == "genbio"
    assert registry["GenBio-PathFM"].model_id == "genbio-ai/genbio-pathfm"


@pytest.mark.parametrize(
    ("name", "model_id", "revision", "embedding_dim"),
    [
        (
            "Mascaret",
            "wearewaiv/mascaret",
            "e95e7ea15e039e78d74def101415e19d9a67ba80",
            1536,
        ),
        (
            "Phaet",
            "wearewaiv/phaet",
            "e0ce6e0ee248470bd8604823e412ca64048a2495",
            1024,
        ),
    ],
)
def test_registry_includes_pinned_fp32_waiv_models(
    name: str, model_id: str, revision: str, embedding_dim: int
) -> None:
    spec = mr._build_model_registry()[name]

    assert spec == mr.ModelSpec(
        backend="waiv",
        model_id=model_id,
        extract="raw",
        mixed_precision=False,
        checkpoint_revision=revision,
        embedding_dim=embedding_dim,
    )


@pytest.mark.parametrize("name", ["Mascaret", "Phaet"])
def test_waiv_loader_implements_the_released_embedding_contract(
    monkeypatch: pytest.MonkeyPatch, name: str, extraction_module
) -> None:
    ee = extraction_module
    spec = mr._build_model_registry()[name]
    expected_embedding = object()
    batch = object()
    calls: list[tuple] = []

    class FakeModel:
        config = types.SimpleNamespace(
            pixel_mean=[0.25, 0.5, 0.75],
            pixel_std=[0.5, 0.25, 0.25],
        )

        def eval(self):
            calls.append(("eval",))
            return self

        def to(self, device):
            calls.append(("to", device))
            return self

        def encode(self, pixel_values):
            calls.append(("encode", pixel_values))
            return expected_embedding

    fake_model = FakeModel()
    monkeypatch.setattr(
        ee,
        "AutoModel",
        types.SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.append(("load", model_id, kwargs)) or fake_model
            )
        ),
    )

    from torchvision.transforms import v2

    monkeypatch.setattr(v2, "Compose", lambda steps: steps)
    monkeypatch.setattr(v2, "ToImage", lambda: ("to_image",))
    monkeypatch.setattr(v2, "Resize", lambda size: ("resize", size))
    monkeypatch.setattr(v2, "CenterCrop", lambda size: ("center_crop", size))
    monkeypatch.setattr(v2, "ToDtype", lambda dtype, *, scale: ("to_dtype", dtype, scale))
    monkeypatch.setattr(v2, "Normalize", lambda *, mean, std: ("normalize", mean, std))

    model, transform, embed_fn = ee._load_model_and_transform(spec, "cpu")
    embedding = embed_fn(batch)

    assert model is fake_model
    assert transform == [
        ("to_image",),
        ("resize", 224),
        ("center_crop", 224),
        ("to_dtype", ee.torch.float32, True),
        ("normalize", [0.25, 0.5, 0.75], [0.5, 0.25, 0.25]),
    ]
    assert embedding is expected_embedding
    assert calls == [
        (
            "load",
            spec.model_id,
            {
                "trust_remote_code": True,
                "revision": spec.checkpoint_revision,
            },
        ),
        ("eval",),
        ("to", "cpu"),
        ("encode", batch),
    ]


@pytest.mark.parametrize(("name", "embedding_dim"), [("Mascaret", 1536), ("Phaet", 1024)])
def test_waiv_embedding_extraction_publishes_finite_fp32_vectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    embedding_dim: int,
    extraction_module,
) -> None:
    ee = extraction_module
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / f"{name}.npy"
    pd.DataFrame(
        {
            "sample_id": ["first", "second"],
            "image_path": ["first.png", "second.png"],
            "label": ["a", "b"],
            "confounder": ["x", "y"],
            "group_id": ["first", "second"],
        }
    ).to_csv(manifest_path, index=False)
    upstream = np.zeros((2, embedding_dim), dtype=np.float64)
    upstream[0, 0] = 1.0
    upstream[1, 1] = 1.0

    class FakeTensor:
        def __init__(self, values: np.ndarray):
            self.values = values

        def __len__(self):
            return len(self.values)

        def to(self, device, non_blocking=False):
            return self

        def float(self):
            return FakeTensor(self.values.astype(np.float32))

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeModel:
        config = types.SimpleNamespace(
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
        )

        def eval(self):
            return self

        def to(self, device):
            return self

        def encode(self, batch):
            return FakeTensor(upstream)

    monkeypatch.setattr(ee.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        ee,
        "DataLoader",
        lambda *args, **kwargs: [FakeTensor(np.zeros((2, 3, 224, 224)))],
    )
    monkeypatch.setattr(ee.torch, "inference_mode", contextlib.nullcontext, raising=False)

    saved_path, shape = ee.embed_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        spec=mr._build_model_registry()[name],
        batch_size=2,
        num_workers=0,
        device_arg="cpu",
        progress_enabled=False,
    )

    embeddings = np.load(saved_path)
    assert shape == (2, embedding_dim)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_array_equal(np.linalg.norm(embeddings, axis=1), [1.0, 1.0])


def test_parse_models_rejects_empty_and_duplicates() -> None:
    with pytest.raises(ValueError, match="empty"):
        mr._parse_models("UNI,,Phikon")
    with pytest.raises(ValueError, match="duplicate"):
        mr._parse_models("UNI,UNI")


def test_unwrap_gpfm_state_dict_handles_wrappers_and_prefixes(extraction_module) -> None:
    ee = extraction_module
    bare = {"cls_token": 1, "blocks.0.norm.weight": 2}
    assert ee._unwrap_gpfm_state_dict(dict(bare)) == bare

    wrapped = {"teacher": {"module.cls_token": 1, "backbone.blocks.0.w": 2}}
    assert ee._unwrap_gpfm_state_dict(wrapped) == {"cls_token": 1, "blocks.0.w": 2}

    nested = {"state_dict": {"backbone.pos_embed": 3}}
    assert ee._unwrap_gpfm_state_dict(nested) == {"pos_embed": 3}


def test_unwrap_gpfm_state_dict_rejects_non_mapping(extraction_module) -> None:
    ee = extraction_module
    with pytest.raises(ValueError, match="state_dict mapping"):
        ee._unwrap_gpfm_state_dict(["not", "a", "state", "dict"])


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_hf_auto_loader_forwards_only_explicit_checkpoint_revision(
    monkeypatch: pytest.MonkeyPatch, revision: str | None, extraction_module
) -> None:
    ee = extraction_module
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        ee,
        "AutoImageProcessor",
        types.SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.append(("processor", kwargs)) or object()
            )
        ),
    )
    monkeypatch.setattr(
        ee,
        "AutoModel",
        types.SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.append(("model", kwargs)) or FakeModel()
            )
        ),
    )
    spec = mr.ModelSpec(
        backend="hf_auto",
        model_id="owner/model",
        checkpoint_revision=revision,
    )

    ee._load_model_and_transform(spec, ee.torch.device("cpu"))

    expected_kwargs = {"trust_remote_code": True}
    if revision is not None:
        expected_kwargs["revision"] = revision
    assert calls == [
        ("processor", expected_kwargs),
        ("model", expected_kwargs),
    ]


@pytest.mark.parametrize(
    ("revision", "expected_model_id"),
    [
        (None, "hf-hub:owner/model"),
        ("a" * 40, f"hf-hub:owner/model@{'a' * 40}"),
    ],
)
def test_timm_loader_forwards_only_explicit_checkpoint_revision(
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
    expected_model_id: str,
    extraction_module,
) -> None:
    ee = extraction_module
    calls: list[str] = []

    class FakeModel:
        pretrained_cfg: dict = {}

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        ee.timm,
        "create_model",
        lambda model_id, pretrained, **kwargs: calls.append(model_id) or FakeModel(),
    )
    monkeypatch.setattr(ee, "resolve_data_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(ee, "create_transform", lambda **kwargs: object())
    spec = mr.ModelSpec(
        backend="timm",
        model_id="hf-hub:owner/model",
        checkpoint_revision=revision,
    )

    ee._load_model_and_transform(spec, ee.torch.device("cpu"))

    assert calls == [expected_model_id]


def test_conch_loader_downloads_pinned_checkpoint_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extraction_module
) -> None:
    ee = extraction_module
    import conch.open_clip_custom
    import huggingface_hub

    calls: list[tuple] = []
    checkpoint = tmp_path / "pytorch_model.bin"

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: calls.append(("download", kwargs)) or str(checkpoint),
    )
    monkeypatch.setattr(
        conch.open_clip_custom,
        "create_model_from_pretrained",
        lambda architecture, path: (
            calls.append(("load", architecture, path)) or (FakeModel(), object())
        ),
    )
    spec = mr.ModelSpec(
        backend="conch_v1",
        model_id="MahmoodLab/conch",
        checkpoint_revision="a" * 40,
        extract="raw",
    )

    ee._load_model_and_transform(spec, ee.torch.device("cpu"))

    assert calls == [
        (
            "download",
            {
                "repo_id": "MahmoodLab/conch",
                "filename": "pytorch_model.bin",
                "revision": "a" * 40,
            },
        ),
        ("load", "conch_ViT-B-16", str(checkpoint)),
    ]


def test_musk_loader_downloads_pinned_checkpoint_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extraction_module
) -> None:
    ee = extraction_module
    import huggingface_hub
    from musk import utils
    from timm import models

    calls: list[tuple] = []
    checkpoint = tmp_path / "model.safetensors"

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(models, "create_model", lambda name: FakeModel())
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: calls.append(("download", kwargs)) or str(checkpoint),
    )
    monkeypatch.setattr(
        utils,
        "load_model_and_may_interpolate",
        lambda path, model, key, prefix: calls.append(("load", path, key, prefix)),
    )
    spec = mr.ModelSpec(
        backend="musk",
        model_id="hf_hub:xiangjx/musk",
        checkpoint_revision="a" * 40,
        extract="ms_aug",
    )

    ee._load_model_and_transform(spec, ee.torch.device("cpu"))

    assert calls == [
        (
            "download",
            {
                "repo_id": "xiangjx/musk",
                "filename": "model.safetensors",
                "revision": "a" * 40,
            },
        ),
        ("load", str(checkpoint), "model|module", ""),
    ]
