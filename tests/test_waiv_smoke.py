"""Opt-in smoke checks against Mascaret and Phaet's real gated weights."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CROMA_RUN_WAIV_SMOKE") != "1",
    reason="set CROMA_RUN_WAIV_SMOKE=1 to download and validate Waiv weights",
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.mark.parametrize(
    ("name", "pooling", "embedding_dim", "unit_normalized"),
    [
        ("Mascaret", "canonical", 1536, True),
        ("Mascaret", "cls-mean-patch", 3072, False),
        ("Phaet", "canonical", 1024, True),
        ("Phaet", "cls-mean-patch", 2048, False),
    ],
)
def test_real_waiv_weights_return_stable_finite_fp32_embeddings(
    name: str, pooling: str, embedding_dim: int, unit_normalized: bool
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    image_module = pytest.importorskip("PIL.Image")
    if int(transformers.__version__.split(".", maxsplit=1)[0]) != 5:
        pytest.skip("Waiv real-weight smoke checks require Transformers 5")

    import extract_embeddings as ee
    from model_registry import _build_model_registry

    device = torch.device(os.environ.get("CROMA_WAIV_SMOKE_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("requested Waiv CUDA smoke check but CUDA is unavailable")
    model, transform, embed_fn = ee._load_model_and_transform(
        _build_model_registry()[name], device, pooling=pooling
    )
    image = image_module.fromarray(np.zeros((112, 224, 3), dtype=np.uint8))
    batch = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        first = embed_fn(batch)
        second = embed_fn(batch)

    assert model.training is False
    assert {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    } == {torch.float32}
    assert batch.shape == (1, 3, 224, 224)
    assert batch.dtype == torch.float32
    assert first.shape == (1, embedding_dim)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if unit_normalized:
        torch.testing.assert_close(
            torch.linalg.vector_norm(first, dim=1),
            torch.ones(1, device=first.device),
            rtol=1e-5,
            atol=1e-6,
        )
