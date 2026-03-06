
import numpy as np
import pandas as pd
import pytest

from mari import CCRR
from mari.metrics.tail import TailMetrics, compute_tail_metrics, select_exact_size_tail_set


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


class TestCCRRTailIntegration:

    def test_ccrr_result_has_tail_fields(self) -> None:
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

        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
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

        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            alpha=0.25,
        )

        assert result.alpha == 0.25
        assert result.sample_values_aligned.shape == (len(manifest),)


class TestExactSizeTailSet:

    def test_selects_exactly_ceil_alpha_defined_rows(self) -> None:
        table = pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2", "s3", "s4"],
                "ccrr_m1": [0.50, 0.20, np.nan, 0.10, 0.40],
            }
        )

        tail = select_exact_size_tail_set(table, value_column="ccrr_m1", alpha=0.40)

        assert tail["sample_id"].tolist() == ["s3", "s1"]
        assert tail["ccrr_m1"].tolist() == [0.10, 0.20]

    def test_excludes_undefined_rows_before_sizing_tail(self) -> None:
        table = pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2", "s3"],
                "ccrr_m1": [np.nan, 0.10, 0.20, np.nan],
            }
        )

        tail = select_exact_size_tail_set(table, value_column="ccrr_m1", alpha=0.50)

        assert tail["sample_id"].tolist() == ["s1"]
        assert tail["ccrr_m1"].tolist() == [0.10]

    def test_breaks_value_ties_by_sample_id(self) -> None:
        table = pd.DataFrame(
            {
                "sample_id": ["s_b", "s_a", "s_c", "s_d"],
                "ccrr_m1": [0.10, 0.10, 0.20, 0.30],
            }
        )

        tail = select_exact_size_tail_set(table, value_column="ccrr_m1", alpha=0.25)

        assert tail["sample_id"].tolist() == ["s_a"]
        assert tail["ccrr_m1"].tolist() == [0.10]

    def test_is_stable_under_input_reordering(self) -> None:
        table = pd.DataFrame(
            {
                "sample_id": ["s_c", "s_a", "s_b", "s_d"],
                "ccrr_m1": [0.10, 0.10, 0.10, 0.20],
            }
        )
        shuffled = table.iloc[[2, 0, 3, 1]].reset_index(drop=True)

        tail_a = select_exact_size_tail_set(table, value_column="ccrr_m1", alpha=0.50)
        tail_b = select_exact_size_tail_set(shuffled, value_column="ccrr_m1", alpha=0.50)

        assert tail_a["sample_id"].tolist() == ["s_a", "s_b"]
        assert tail_b["sample_id"].tolist() == ["s_a", "s_b"]
