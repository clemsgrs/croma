import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "bench" / "extract_embeddings.py"
WAIV_SMOKE_PATH = ROOT / "tests" / "test_waiv_smoke.py"


def test_module_imports_without_bench_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        root = str(name).split(".")[0]
        if root in {"torch", "torchvision", "timm", "transformers", "PIL", "slide2vec"}:
            raise ModuleNotFoundError(f"No module named '{root}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "extract_embeddings_no_deps"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)


def test_waiv_reproduction_environment_pins_transformers_5_without_slide2vec() -> None:
    requirements = (ROOT / "scripts" / "bench" / "requirements-waiv.txt").read_text(
        encoding="utf-8"
    )
    dependencies = [
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "transformers>=5.14,<6" in dependencies
    assert all(not dependency.startswith("slide2vec") for dependency in dependencies)


def test_waiv_smoke_collects_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        root = str(name).split(".")[0]
        if root in {"torch", "torchvision", "transformers", "PIL"}:
            raise ModuleNotFoundError(f"No module named '{root}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "test_waiv_smoke_no_deps"
    spec = importlib.util.spec_from_file_location(module_name, WAIV_SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
