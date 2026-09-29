"""Opt-in real-weight smoke test for the pinned Mettle checkpoint."""

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


@pytest.mark.skipif(
    os.environ.get("CROMA_RUN_METTLE_SMOKE") != "1",
    reason="set CROMA_RUN_METTLE_SMOKE=1 to download and run the Mettle checkpoint",
)
def test_real_mettle_cls_mean_is_exact_finite_and_deterministic() -> None:
    try:
        import torch
        from PIL import Image
    except ModuleNotFoundError as exc:
        pytest.fail(f"Mettle smoke dependencies are missing: {exc}")

    import extract_embeddings as ee
    import model_registry as mr

    spec = mr._build_model_registry()["Mettle"]
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

    assert batch.shape == (2, 3, 224, 224)
    assert first.shape == (2, 3072)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[:, :1536], cls, rtol=0, atol=0)
    assert torch.equal(first, second)
    assert model.training is False

    del model, batch, first, second, cls
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
