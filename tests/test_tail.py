import numpy as np
import pandas as pd
import pytest

from croma import CRoMa
from croma.metrics.tail import TailMetrics, compute_tail_metrics


class TestComputeTailMetrics:

    def test_known_values(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = compute_tail_metrics(values, alpha=0.10)

        assert result.alpha == 0.10
        assert result.q_alpha == pytest.approx(1.9)
        tail = values[values <= result.q_alpha]
        assert result.ltm_alpha == pytest.approx(float(tail.mean()))
        assert result.n_tail_samples == len(tail)

    def test_empty_array(self) -> None:
        result = compute_tail_metrics(np.array([]), alpha=0.10)

        assert result.alpha == 0.10
        assert np.isnan(result.q_alpha)
        assert np.isnan(result.ltm_alpha)
        assert result.n_tail_samples == 0

    def test_all_equal(self) -> None:
        values = np.full(20, 5.0)
        result = compute_tail_metrics(values, alpha=0.10)

        assert result.q_alpha == pytest.approx(5.0)
        assert result.ltm_alpha == pytest.approx(5.0)

    def test_alpha_half(self) -> None:
        values = np.arange(1.0, 101.0)
        result = compute_tail_metrics(values, alpha=0.50)

        assert result.alpha == 0.50
        assert result.q_alpha == pytest.approx(50.5)
        tail = values[values <= result.q_alpha]
        assert result.ltm_alpha == pytest.approx(float(tail.mean()))
        assert result.n_tail_samples == 50

    def test_ignores_nan_values(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        result = compute_tail_metrics(values, alpha=0.20)

        # NaN is filtered out, so only [1, 2, 3, 4, 5] are used
        finite = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_q = float(np.percentile(finite, 20))
        assert result.q_alpha == pytest.approx(expected_q)

    def test_returns_frozen_dataclass(self) -> None:
        result = compute_tail_metrics(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, TailMetrics)
        with pytest.raises(AttributeError):
            result.alpha = 0.5  # type: ignore[misc]


class TestCRoMaTailIntegration:

    def test_croma_result_has_tail_fields(self) -> None:
        manifest = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(8)],
                "image_path": [f"/tmp/{i}.png" for i in range(8)],
                "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "scanner_vendor": ["V1", "V1", "V2", "V2", "V1", "V1", "V2", "V2"],
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

        result = CRoMa.compute(
            features=features,
            manifest=manifest,
            confounder_column="scanner_vendor",
            evaluation_design="dataset_wide",
            m=1,
            alpha=0.10,
        )

        assert result.alpha == 0.10
        assert np.isfinite(result.q_alpha)
        assert np.isfinite(result.ltm_alpha)
        assert result.ltm_alpha <= result.q_alpha
        assert result.sample_values_aligned.shape == (len(manifest),)

    def test_custom_alpha(self) -> None:
        manifest = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(8)],
                "image_path": [f"/tmp/{i}.png" for i in range(8)],
                "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "scanner_vendor": ["V1", "V1", "V2", "V2", "V1", "V1", "V2", "V2"],
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

        result = CRoMa.compute(
            features=features,
            manifest=manifest,
            confounder_column="scanner_vendor",
            evaluation_design="dataset_wide",
            m=1,
            alpha=0.25,
        )

        assert result.alpha == 0.25
        assert result.sample_values_aligned.shape == (len(manifest),)
