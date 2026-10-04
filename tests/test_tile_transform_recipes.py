"""Six tile encoders are preprocessed with their authors' recipe, as slide2vec does.

timm's hub config for these checkpoints holds timm defaults or the checkpoint's native
resolution, not the recipe the authors evaluate with, so each gets an explicit recipe
instead of ``resolve_data_config``. The remaining tile encoders keep their transform.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_registry as mr

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HOPTIMUS_MEAN = [0.707223, 0.578729, 0.703617]
HOPTIMUS_STD = [0.211883, 0.230117, 0.177517]

# The target column of the table in the issue (slide2vec >= 6.2 recipes).
AUTHORS_RECIPES = {
    "DINOv2-B": {
        "resize": 256,
        "center_crop": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "H-optimus-0": {
        "resize": 224,
        "center_crop": 224,
        "mean": HOPTIMUS_MEAN,
        "std": HOPTIMUS_STD,
    },
    "H-optimus-1": {
        "resize": 224,
        "center_crop": 224,
        "mean": HOPTIMUS_MEAN,
        "std": HOPTIMUS_STD,
    },
    "Prov-GigaPath": {
        "resize": 256,
        "center_crop": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "GPFM": {
        "resize": [224, 224],
        "center_crop": None,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "mSTAR": {
        "resize": [224, 224],
        "center_crop": None,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
}


# The tile encoders whose transform already matches slide2vec's and must not move.
ALIGNED_MODELS = {
    "Virchow": "timm",
    "Virchow2": "timm",
    "UNI": "timm",
    "UNI2-h": "timm",
    "H0-mini": "timm",
    "Prost40M": "timm",
    "RudolfV 2": "rudolfv2",
    "RudolfV 2-B": "rudolfv2",
    "RudolfV 2-S": "rudolfv2",
    "Mettle": "mettle",
    "Midnight-12k": "midnight",
    "CONCH": "conch_v1",
    "CONCHv1.5": "conch_v1_5",
    "MUSK": "musk",
    "GenBio-PathFM": "genbio",
    "Phikon": "hf_auto",
    "Phikon-v2": "hf_auto",
    "Hibou-B": "hf_auto",
    "Hibou-L": "hf_auto",
    "Mascaret": "waiv",
    "Phaet": "waiv",
}
# Backends whose recorded contract already carries their own preprocessing.
BACKENDS_WITH_OWN_CONTRACT = {"rudolfv2", "mettle", "waiv"}


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["a.png", "b.png"],
            "label": ["x", "y"],
            "confounder": ["c", "d"],
            "group_id": ["a", "b"],
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("name", sorted(AUTHORS_RECIPES))
def test_artifact_contract_records_the_authors_recipe(
    name: str, tmp_path: Path, extraction_module
) -> None:
    target = AUTHORS_RECIPES[name]

    contract = extraction_module.build_embedding_artifact_contract(
        manifest_path=_manifest(tmp_path),
        spec=mr._build_model_registry()[name],
        batch_size=64,
        device_arg="cpu",
    )

    preprocessing = contract.extraction_contract["preprocessing"]
    assert {
        "resize": preprocessing["resize"],
        "resize_interpolation": preprocessing["resize_interpolation"],
        "center_crop": preprocessing["center_crop"],
        "normalization_mean": preprocessing["normalization_mean"],
        "normalization_std": preprocessing["normalization_std"],
    } == {
        "resize": target["resize"],
        "resize_interpolation": "bicubic",
        "center_crop": target["center_crop"],
        "normalization_mean": target["mean"],
        "normalization_std": target["std"],
    }


def _expected_operations(name: str, float32) -> list[tuple]:
    """The authors' transform as slide2vec composes it, op by op."""
    target = AUTHORS_RECIPES[name]
    resize = (
        "resize",
        target["resize"] if isinstance(target["resize"], int) else tuple(target["resize"]),
        {"interpolation": "bicubic", "antialias": True},
    )
    if target["center_crop"] is None:
        # GPFM and mSTAR: torchvision Resize on the PIL tile, then ToTensor.
        head = [resize, ("to_image",)]
    else:
        head = [("to_image",), resize, ("center_crop", target["center_crop"])]
    return head + [
        ("to_dtype", float32, {"scale": True}),
        ("normalize", {"mean": tuple(target["mean"]), "std": tuple(target["std"])}),
    ]


def _record_v2_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    v2 = sys.modules["torchvision.transforms.v2"]
    monkeypatch.setattr(v2, "Compose", lambda operations: operations)
    monkeypatch.setattr(v2, "ToImage", lambda: ("to_image",))
    monkeypatch.setattr(v2, "Resize", lambda size, **kwargs: ("resize", size, kwargs))
    monkeypatch.setattr(v2, "CenterCrop", lambda size: ("center_crop", size))
    monkeypatch.setattr(v2, "ToDtype", lambda dtype, **kwargs: ("to_dtype", dtype, kwargs))
    monkeypatch.setattr(
        v2,
        "Normalize",
        lambda **kwargs: (
            "normalize",
            {key: tuple(value) for key, value in kwargs.items()},
        ),
    )


class _FakeTimmModel:
    pretrained_cfg: dict = {}

    def eval(self):
        return self

    def to(self, device):
        return self

    def load_state_dict(self, state, strict):
        return None


def _patch_checkpoint_loading(monkeypatch: pytest.MonkeyPatch, ee) -> None:
    """Stand in for every weight download/load so no checkpoint is ever fetched."""
    monkeypatch.setattr(ee.timm, "create_model", lambda *args, **kwargs: _FakeTimmModel())
    monkeypatch.setattr(
        sys.modules["huggingface_hub"],
        "hf_hub_download",
        lambda **kwargs: "GPFM.pth",
    )
    monkeypatch.setattr(ee.torch, "load", lambda *args, **kwargs: {}, raising=False)


@pytest.mark.parametrize("name", sorted(AUTHORS_RECIPES))
def test_loader_builds_the_authors_transform_not_the_hub_config(
    name: str, monkeypatch: pytest.MonkeyPatch, extraction_module
) -> None:
    # resolve_data_config / create_transform stay as the fixture's tripwires.
    ee = extraction_module
    _record_v2_operations(monkeypatch)
    _patch_checkpoint_loading(monkeypatch, ee)

    _model, transform, _embed = ee._load_model_and_transform(
        mr._build_model_registry()[name], "cpu"
    )

    assert transform == _expected_operations(name, ee.torch.float32)


@pytest.fixture(scope="module")
def real_extraction_module():
    """The extraction script against the real torch/torchvision (no weights loaded)."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    module_name = "extract_embeddings_with_real_transforms"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / "extract_embeddings.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("name", sorted(AUTHORS_RECIPES))
def test_a_256_px_tile_becomes_a_3x224x224_tensor(name: str, real_extraction_module) -> None:
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    ee = real_extraction_module
    recipe = ee._tile_transform_recipe(mr._build_model_registry()[name])
    tile = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8))

    tensor = recipe.build_transform(v2)(tile)

    assert (tuple(tensor.shape), tensor.dtype) == ((3, 224, 224), torch.float32)


def test_the_six_recipes_and_the_aligned_models_cover_the_registry() -> None:
    registry = mr._build_model_registry()

    assert sorted(registry) == sorted([*AUTHORS_RECIPES, *ALIGNED_MODELS])
    assert {name: registry[name].backend for name in ALIGNED_MODELS} == ALIGNED_MODELS


@pytest.mark.parametrize("name", sorted(ALIGNED_MODELS))
def test_aligned_models_keep_their_recorded_contract(
    name: str, tmp_path: Path, extraction_module
) -> None:
    ee = extraction_module
    spec = mr._build_model_registry()[name]

    contract = ee.build_embedding_artifact_contract(
        manifest_path=_manifest(tmp_path),
        spec=spec,
        batch_size=64,
        device_arg="cpu",
    )

    assert ee._tile_transform_recipe(spec) is None
    preprocessing = contract.extraction_contract.get("preprocessing")
    if spec.backend in BACKENDS_WITH_OWN_CONTRACT:
        # Their exact preprocessing is pinned by test_extract_embeddings_registry.py.
        assert preprocessing.get("source") != "authors-recipe"
    else:
        # Nothing recorded, exactly as before: their existing artifacts stay reusable.
        assert preprocessing is None


@pytest.mark.parametrize(
    "name", sorted(name for name, backend in ALIGNED_MODELS.items() if backend == "timm")
)
def test_aligned_timm_models_still_resolve_the_checkpoint_hub_config(
    name: str, monkeypatch: pytest.MonkeyPatch, extraction_module
) -> None:
    ee = extraction_module
    hub_cfg = {"input_size": (3, 224, 224)}
    hub_transform = object()
    resolved: list[dict] = []

    class FakeModel(_FakeTimmModel):
        pretrained_cfg = hub_cfg

    monkeypatch.setattr(ee.timm, "create_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        ee,
        "resolve_data_config",
        lambda cfg, model: resolved.append(cfg) or {"resolved": cfg},
    )
    monkeypatch.setattr(
        ee,
        "create_transform",
        lambda **kwargs: hub_transform if kwargs == {"resolved": hub_cfg} else None,
    )

    _model, transform, _embed = ee._load_model_and_transform(
        mr._build_model_registry()[name], "cpu"
    )

    assert (transform, resolved) == (hub_transform, [hub_cfg])
