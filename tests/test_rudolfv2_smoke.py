"""Opt-in real-weight smoke tests for the pinned RudolfV 2 family."""

import gc
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.mark.skipif(
    os.environ.get("CROMA_RUN_RUDOLFV2_SMOKE") != "1",
    reason="set CROMA_RUN_RUDOLFV2_SMOKE=1 to run gated RudolfV 2 checkpoints",
)
@pytest.mark.parametrize(
    ("name", "model_width", "pooled_width"),
    [
        ("RudolfV 2", 1536, 3072),
        ("RudolfV 2-B", 768, 1536),
        ("RudolfV 2-S", 384, 768),
    ],
)
def test_real_rudolfv2_native_pooling_is_exact_and_deterministic(
    name: str, model_width: int, pooled_width: int
) -> None:
    try:
        import timm  # noqa: F401 - required by the pinned remote code
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        pytest.fail(f"RudolfV 2 smoke dependencies are missing: {exc}")

    import extract_embeddings as ee
    import model_registry as mr

    assert transformers.__version__.split(".", maxsplit=1)[0] == "5"
    spec = mr._build_model_registry()[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _transform, embed = ee._load_model_and_transform(spec, device)
    batch = torch.linspace(
        -1.0,
        1.0,
        3 * 224 * 224,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 224, 224)

    with torch.inference_mode():
        published = model.model.encode(batch)
        first = embed(batch)
        second = embed(batch)
    expected = torch.cat(
        [
            published["x_norm_clstoken"],
            published["x_norm_patchtokens"].mean(1),
        ],
        dim=-1,
    )

    assert published["last_hidden_state"].shape == (1, 793, model_width)
    assert first.shape == (1, pooled_width)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert torch.equal(first, second)
    assert model.training is False

    del model, batch, published, first, second, expected
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
