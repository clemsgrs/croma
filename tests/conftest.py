"""Shared harness for driving the compute-only benchmark driver in tests.

``benchmark.py`` resolves everything from two places: the benchmark registry and the
output layout. So a test sets up a *tileset* on disk (embeddings + manifest.csv),
registers a *benchmark* that views it, and points ``layout`` at the temp tree.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmarks as benchmarks_module  # noqa: E402
import layout as layout_module  # noqa: E402
from benchmarks import BenchmarkSpec  # noqa: E402


@pytest.fixture
def extraction_module(monkeypatch):
    """Load the extraction script against deterministic encoder-boundary fakes."""

    # The extraction script imports croma before its optional encoder stack. Mirror that
    # order so scipy/sklearn see an absent (rather than deliberately fake) torch module.
    import croma  # noqa: F401

    def unexpected_boundary_call(*args, **kwargs):
        raise AssertionError("optional encoder boundary was not configured by the test")

    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.nn = types.SimpleNamespace(SiLU=object())
    torch.float16 = object()
    torch.float32 = object()
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = unexpected_boundary_call
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils

    timm = types.ModuleType("timm")
    timm.create_model = unexpected_boundary_call
    timm_data = types.ModuleType("timm.data")
    timm_data.resolve_data_config = unexpected_boundary_call
    timm_transforms = types.ModuleType("timm.data.transforms_factory")
    timm_transforms.create_transform = unexpected_boundary_call
    timm_layers = types.ModuleType("timm.layers")
    timm_layers.SwiGLUPacked = object()
    timm_models = types.ModuleType("timm.models")
    timm_models.create_model = unexpected_boundary_call
    timm.data = timm_data
    timm.layers = timm_layers
    timm.models = timm_models

    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = types.SimpleNamespace(
        from_pretrained=unexpected_boundary_call
    )
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=unexpected_boundary_call)

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = unexpected_boundary_call

    conch = types.ModuleType("conch")
    conch_open_clip = types.ModuleType("conch.open_clip_custom")
    conch_open_clip.create_model_from_pretrained = unexpected_boundary_call
    conch.open_clip_custom = conch_open_clip

    musk = types.ModuleType("musk")
    musk_modeling = types.ModuleType("musk.modeling")
    musk_utils = types.ModuleType("musk.utils")
    musk_utils.load_model_and_may_interpolate = unexpected_boundary_call
    musk.modeling = musk_modeling
    musk.utils = musk_utils

    transforms_v2 = types.ModuleType("torchvision.transforms.v2")
    transforms_v2.Compose = lambda operations: operations
    transforms_v2.ToImage = lambda: object()
    transforms_v2.Resize = lambda *args, **kwargs: object()
    transforms_v2.CenterCrop = lambda *args, **kwargs: object()
    transforms_v2.ToDtype = lambda *args, **kwargs: object()
    transforms_v2.Normalize = lambda *args, **kwargs: object()
    transforms_v2.InterpolationMode = types.SimpleNamespace(BICUBIC="bicubic")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_transforms.v2 = transforms_v2
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = torchvision_transforms

    fake_modules = {
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "timm": timm,
        "timm.data": timm_data,
        "timm.data.transforms_factory": timm_transforms,
        "timm.layers": timm_layers,
        "timm.models": timm_models,
        "transformers": transformers,
        "huggingface_hub": huggingface_hub,
        "conch": conch,
        "conch.open_clip_custom": conch_open_clip,
        "musk": musk,
        "musk.modeling": musk_modeling,
        "musk.utils": musk_utils,
        "torchvision": torchvision,
        "torchvision.transforms": torchvision_transforms,
        "torchvision.transforms.v2": transforms_v2,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "extract_embeddings_with_fake_boundaries"
    script_path = SCRIPTS / "extract_embeddings.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bench_env(tmp_path, monkeypatch):
    """Retarget layout at a temp tree and give a builder for tilesets/benchmarks."""
    monkeypatch.setattr(layout_module, "REPO", tmp_path)
    monkeypatch.setattr(layout_module, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(benchmarks_module, "BENCHMARKS", dict(benchmarks_module.BENCHMARKS))
    return _BenchEnv(tmp_path, monkeypatch)


class _BenchEnv:
    def __init__(self, root: Path, monkeypatch) -> None:
        self.root = root
        self._monkeypatch = monkeypatch

    def write_tileset(
        self,
        tileset: str,
        manifest: pd.DataFrame,
        features: dict[str, np.ndarray],
    ) -> Path:
        """Materialise ``output/embeddings/<tileset>/`` with manifest.csv + <Model>.npy."""
        directory = layout_module.embeddings_dir(tileset)
        directory.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(layout_module.tileset_manifest(tileset), index=False)
        for model, matrix in features.items():
            arr = np.asarray(matrix, dtype=np.float32)
            if arr.shape[0] != len(manifest):
                raise AssertionError(
                    f"{model}: {arr.shape[0]} embedding rows vs {len(manifest)} manifest rows"
                )
            np.save(directory / f"{model}.npy", arr)
        return directory

    def register(
        self,
        name: str,
        *,
        tileset: str,
        manifest: pd.DataFrame,
        design: str = "all",
        k_max: int = 5,
        confounder_column: str = "medical_center",
    ) -> BenchmarkSpec:
        """Write a benchmark's eval manifest under data/ and register it."""
        rel = Path("data") / f"{name}.csv"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(target, index=False)
        spec = BenchmarkSpec(
            name=name,
            tileset=tileset,
            manifest=str(rel),
            design=design,
            k_max=k_max,
            confounder_column=confounder_column,
        )
        benchmarks_module.BENCHMARKS[name] = spec
        return spec

    def respec(self, name: str, **changes) -> BenchmarkSpec:
        spec = replace(benchmarks_module.BENCHMARKS[name], **changes)
        benchmarks_module.BENCHMARKS[name] = spec
        return spec

    def run(self, benchmark: str, protocol: str = "k-star", *extra: str) -> int:
        import benchmark as bm

        argv = ["benchmark.py", "--benchmark", benchmark, "--protocol", protocol, *extra]
        self._monkeypatch.setattr(sys, "argv", argv)
        return bm.main()

    def results_dir(self, benchmark: str, protocol: str = "k-star") -> Path:
        return layout_module.results_dir(protocol, benchmark)
