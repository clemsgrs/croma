import numpy as np
import pytest

from croma import nipd


def test_nipd_integrates_normalized_degradation_over_an_irregular_v_grid() -> None:
    # Baseline skill is 0.40. The curve retains fractions 1.0, 0.8 and 0.0 of that
    # skill at V = 0, 0.2 and 1, so normalized degradation is g(V) = -V. Its area
    # over [0, 1] is exactly -0.5, irrespective of the irregular middle point.
    accuracies = np.array(
        [
            [0.90, 0.90],
            [0.82, 0.82],
            [0.50, 0.50],
        ],
        dtype=float,
    )

    result = nipd(accuracies, cramers_v=[0.0, 0.2, 1.0], chance=0.5)

    assert result == pytest.approx(-0.5)


def test_nipd_rejects_a_v_grid_that_does_not_match_the_accuracy_rows() -> None:
    accuracies = np.array(
        [
            [0.90, 0.90],
            [0.75, 0.75],
            [0.60, 0.60],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="one Cramér's-V value per accuracy row"):
        nipd(accuracies, cramers_v=[0.0, 1.0], chance=0.5)


@pytest.mark.parametrize(
    "cramers_v",
    [
        [0.0, np.nan, 1.0],
        [0.0, 0.8, 0.4],
        [0.0, 0.4, 0.4],
        [0.1, 0.5, 1.0],
        [0.0, 0.5, 0.9],
    ],
)
def test_nipd_requires_a_finite_increasing_v_grid_spanning_zero_to_one(
    cramers_v: list[float],
) -> None:
    accuracies = np.array(
        [[0.90, 0.90], [0.75, 0.75], [0.60, 0.60]],
        dtype=float,
    )

    with pytest.raises(ValueError, match="finite, strictly increasing, and span"):
        nipd(accuracies, cramers_v=cramers_v, chance=0.5)


def test_nipd_is_invariant_to_the_sampling_density_of_a_linear_curve() -> None:
    coarse = np.array([[0.90], [0.50]], dtype=float)
    dense = np.array([[0.90], [0.80], [0.70], [0.60], [0.50]], dtype=float)

    coarse_result = nipd(coarse, cramers_v=[0.0, 1.0], chance=0.5)
    dense_result = nipd(dense, cramers_v=[0.0, 0.25, 0.5, 0.75, 1.0], chance=0.5)

    assert coarse_result == pytest.approx(-0.5)
    assert dense_result == pytest.approx(-0.5)


def test_nipd_aggregates_repeats_before_normalizing_baseline_skill() -> None:
    # Mean baseline is 0.705 and mean V=1 accuracy is 0.660. The endpoint
    # degradation is therefore -0.045 / 0.205 = -9/41, and the straight
    # trapezoid from zero has area -9/82. Normalizing each repeat first would
    # instead produce endpoint changes of -20% and -100% and area -30%.
    accuracies = np.array(
        [
            [0.90, 0.51],
            [0.82, 0.50],
        ],
        dtype=float,
    )

    result = nipd(accuracies, cramers_v=[0.0, 1.0], chance=0.5)

    assert result == pytest.approx(-9 / 82)


@pytest.mark.parametrize(
    "baseline",
    [
        [0.50, 0.50],
        [0.40, 0.40],
    ],
)
def test_nipd_requires_mean_baseline_skill_above_chance(baseline: list[float]) -> None:
    accuracies = np.array([baseline, [0.45, 0.45]], dtype=float)

    with pytest.raises(ValueError, match="above chance"):
        nipd(accuracies, cramers_v=[0.0, 1.0], chance=0.5)
