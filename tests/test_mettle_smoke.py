"""Opt-in real-weight smoke test for the pinned Mettle checkpoints."""

import functools
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytestmark = pytest.mark.skipif(
    os.environ.get("CROMA_RUN_METTLE_SMOKE") != "1",
    reason="set CROMA_RUN_METTLE_SMOKE=1 to download and run the Mettle checkpoints",
)

METTLE_CHECKPOINTS = ("Mettle", "Mettle-1.5")


@functools.cache
def _encode_fixed_batch(name: str) -> dict:
    """Load ``name`` once, encode one fixed batch, and keep the outputs on the CPU."""
    try:
        import torch
        from PIL import Image
    except ModuleNotFoundError as exc:
        pytest.fail(f"Mettle smoke dependencies are missing: {exc}")

    import extract_embeddings as ee
    import model_registry as mr

    spec = mr._build_model_registry()[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Verifies the pinned weights' SHA-256 and the processor contract before loading.
    model, transform, embed = ee._load_model_and_transform(spec, device)

    # Two 256-px tiles, the PathoROB tile size, through the checkpoint's own processor.
    ramp = np.linspace(0, 255, 256 * 256 * 3).reshape(256, 256, 3).astype(np.uint8)
    tiles = [Image.fromarray(ramp), Image.fromarray(ramp[::-1].copy())]
    batch = torch.stack([transform(tile) for tile in tiles]).to(device)

    with torch.inference_mode():
        first = embed(batch)
        second = embed(batch)
        cls = model.encode(batch, feature_view="cls")

    outputs = {
        "batch_shape": tuple(batch.shape),
        "training": model.training,
        "first": first.cpu(),
        "second": second.cpu(),
        "cls": cls.cpu(),
    }
    del model, batch, first, second, cls
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


@pytest.mark.parametrize("name", METTLE_CHECKPOINTS)
def test_real_mettle_cls_mean_is_exact_finite_and_deterministic(name: str) -> None:
    import torch

    outputs = _encode_fixed_batch(name)
    first = outputs["first"]

    assert outputs["batch_shape"] == (2, 3, 224, 224)
    assert first.shape == (2, 3072)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[:, :1536], outputs["cls"], rtol=0, atol=0)
    assert torch.equal(first, outputs["second"])
    assert outputs["training"] is False


def test_real_mettle_checkpoints_encode_the_same_batch_differently() -> None:
    import torch

    mettle_1, mettle_1_5 = (_encode_fixed_batch(name)["first"] for name in METTLE_CHECKPOINTS)

    assert not torch.allclose(mettle_1, mettle_1_5, rtol=1e-3, atol=1e-3)
