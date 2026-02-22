from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mari import MaRI, RI


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("mode", ["paired", "global"])
def test_compute_end_to_end(metric_cls, mode: str) -> None:
    pytest.importorskip("sklearn")

    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )

    features = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.92, 0.08, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00],
            [0.08, 0.92, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00],
        ],
        dtype=float,
    )

    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        mode=mode,
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )

    assert result.dataset == "toy"
    assert result.k in {1, 3}
    assert 0.0 <= result.value <= 1.0
    assert result.n_pairs == 1
    assert result.pair_values.shape == (1,)
    assert result.sample_values.shape[0] == 8


def test_invalid_mode_rejected() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "medical_center": ["C1", "C2"],
            "slide_id": ["slide-0", "slide-1"],
        }
    )
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="mode"):
        RI.compute(
            features=features,
            manifest=manifest,
            mode="auto",
            k_candidates=[1],
        )
