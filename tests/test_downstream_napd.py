import numpy as np
import pytest

from croma import napd


def test_napd_is_zero_when_no_split_degrades() -> None:
    # Every confounded split scores exactly what the balanced baseline scores, so no
    # skill was destroyed and the drop is zero.
    accuracies = np.array(
        [
            [0.80, 0.80],
            [0.80, 0.80],
            [0.80, 0.80],
        ],
        dtype=float,
    )

    assert napd(accuracies, chance=0.5) == pytest.approx(0.0)


def test_napd_is_negative_and_scales_with_the_share_of_skill_lost() -> None:
    # Baseline skill 0.9 - 0.5 = 0.4; both confounded splits retain 0.7 - 0.5 = 0.2,
    # i.e. half the learnable signal survives, so the drop is exactly -0.5.
    accuracies = np.array(
        [
            [0.90, 0.90],
            [0.70, 0.70],
            [0.70, 0.70],
        ],
        dtype=float,
    )

    assert napd(accuracies, chance=0.5) == pytest.approx(-0.5)


def test_napd_orders_two_matrices_by_how_much_skill_they_lose() -> None:
    baseline = [0.90, 0.90]
    mild = np.array([baseline, [0.85, 0.85], [0.85, 0.85]], dtype=float)
    severe = np.array([baseline, [0.60, 0.60], [0.60, 0.60]], dtype=float)

    assert napd(severe, chance=0.5) < napd(mild, chance=0.5) < 0.0


def test_napd_reads_the_same_accuracies_differently_under_each_class_count() -> None:
    # The same accuracy drop destroys a different share of the *learnable* signal
    # depending on how much of the score is free: skill is 0.3/0.55/0.6333 at 2/4/6
    # classes, so the drop shrinks as the class count grows.
    accuracies = np.array(
        [
            [0.80, 0.80],
            [0.70, 0.70],
        ],
        dtype=float,
    )

    binary = napd(accuracies, chance=1 / 2)
    four_class = napd(accuracies, chance=1 / 4)
    six_class = napd(accuracies, chance=1 / 6)

    assert binary == pytest.approx(-1 / 3)
    assert four_class == pytest.approx(-0.1 / 0.55)
    assert six_class == pytest.approx(-0.1 / (0.8 - 1 / 6))
    assert binary < four_class < six_class < 0.0


def test_napd_survives_one_replicate_whose_baseline_dips_toward_chance() -> None:
    # Five healthy replicates plus one where the baseline collapses to 0.52 and the
    # confounded split to 0.56 -- skills of 0.02 and 0.06, whose *ratio* is 3.0. The
    # naive per-replicate reduction averages that ratio in and reports +0.125, i.e. it
    # claims the confounder *helped*. Averaging the replicates first keeps the answer
    # next to the one the healthy replicates alone give.
    healthy = np.array([[0.90] * 5, [0.80] * 5], dtype=float)
    with_dip = np.array([[0.90] * 5 + [0.52], [0.80] * 5 + [0.56]], dtype=float)

    reference = napd(healthy, chance=0.5)
    dipped = napd(with_dip, chance=0.5)

    assert reference == pytest.approx(-0.25)
    assert dipped == pytest.approx(reference, abs=0.03)
    assert dipped < 0.0


def test_napd_returns_a_value_on_a_baseline_that_barely_beats_chance() -> None:
    # 0.55 against binary chance attains a tenth of the achievable headroom, and 0.5001
    # a two-thousandth: both are cells a skill floor would have declared unmeasurable.
    # nAPD carries no floor, so both yield an ordinary finite number.
    faint = np.array([[0.55, 0.55], [0.53, 0.53]], dtype=float)
    fainter = np.array([[0.5001, 0.5001], [0.50005, 0.50005]], dtype=float)

    assert napd(faint, chance=0.5) == pytest.approx(-0.4)
    assert napd(fainter, chance=0.5) == pytest.approx(-0.5)


def test_napd_always_returns_a_finite_float_never_a_sentinel() -> None:
    matrices = [
        np.array([[0.99, 0.99], [0.98, 0.97]], dtype=float),
        np.array([[0.52, 0.51], [0.51, 0.53]], dtype=float),
        np.array([[0.70, 0.90], [0.40, 0.45], [0.95, 0.60]], dtype=float),
    ]

    for accuracies in matrices:
        value = napd(accuracies, chance=0.25)
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_napd_rejects_a_matrix_with_no_confounded_split() -> None:
    # A baseline row on its own averages an empty set of ratios; numpy would answer nan.
    baseline_only = np.array([[0.80, 0.80]], dtype=float)

    with pytest.raises(ValueError, match="confounded split"):
        napd(baseline_only, chance=0.5)


def test_napd_rejects_a_matrix_that_is_not_two_dimensional() -> None:
    one_split_flat = np.array([0.80, 0.70, 0.60], dtype=float)

    with pytest.raises(ValueError, match="2-D"):
        napd(one_split_flat, chance=0.5)


def test_napd_rejects_a_matrix_with_no_replicates() -> None:
    no_iterations = np.zeros((3, 0), dtype=float)

    with pytest.raises(ValueError, match="replicate"):
        napd(no_iterations, chance=0.5)


def test_napd_rejects_a_baseline_that_does_not_beat_chance() -> None:
    # With no skill at the baseline there is no denominator, and below chance every
    # ratio silently inverts its sign -- both are the reduction's domain running out,
    # not a judgement that the cell is too noisy to report.
    at_chance = np.array([[0.50, 0.50], [0.45, 0.45]], dtype=float)
    below_chance = np.array([[0.40, 0.40], [0.45, 0.45]], dtype=float)

    for accuracies in (at_chance, below_chance):
        with pytest.raises(ValueError, match="above chance"):
            napd(accuracies, chance=0.5)


def test_napd_rejects_a_chance_level_outside_the_unit_interval() -> None:
    accuracies = np.array([[0.80, 0.80], [0.70, 0.70]], dtype=float)

    for bad_chance in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="chance"):
            napd(accuracies, chance=bad_chance)


def test_napd_rejects_a_matrix_holding_a_non_finite_score() -> None:
    accuracies = np.array([[0.80, np.nan], [0.70, 0.70]], dtype=float)

    with pytest.raises(ValueError, match="finite"):
        napd(accuracies, chance=0.5)


def test_napd_accepts_a_plain_nested_list() -> None:
    # The signature says ndarray, but anything array-like is coerced -- a caller who
    # assembled the sweep in Python lists should not have to convert it first.
    assert napd([[0.90, 0.90], [0.70, 0.70]], chance=0.5) == pytest.approx(-0.5)
