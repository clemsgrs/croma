import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_embeddings as ee
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


def test_parse_models_rejects_empty_and_duplicates() -> None:
    with pytest.raises(ValueError, match="empty"):
        mr._parse_models("UNI,,Phikon")
    with pytest.raises(ValueError, match="duplicate"):
        mr._parse_models("UNI,UNI")


def test_unwrap_gpfm_state_dict_handles_wrappers_and_prefixes() -> None:
    bare = {"cls_token": 1, "blocks.0.norm.weight": 2}
    assert ee._unwrap_gpfm_state_dict(dict(bare)) == bare

    wrapped = {"teacher": {"module.cls_token": 1, "backbone.blocks.0.w": 2}}
    assert ee._unwrap_gpfm_state_dict(wrapped) == {"cls_token": 1, "blocks.0.w": 2}

    nested = {"state_dict": {"backbone.pos_embed": 3}}
    assert ee._unwrap_gpfm_state_dict(nested) == {"pos_embed": 3}


def test_unwrap_gpfm_state_dict_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="state_dict mapping"):
        ee._unwrap_gpfm_state_dict(["not", "a", "state", "dict"])


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_hf_auto_loader_forwards_only_explicit_checkpoint_revision(
    monkeypatch: pytest.MonkeyPatch, revision: str | None
) -> None:
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
) -> None:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
