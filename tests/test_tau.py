"""Tests for principled MaRI tau assessment and the warn_tau guidance in MaRI.compute."""

import warnings

import numpy as np
import pytest

from croma import MaRI
from croma.metrics.tau import TauAssessment, assess_tau, format_tau_warning
from metric_harness import CONFOUNDER_COLUMN, PINNED_K, constant_embedding

# ---------------------------------------------------------------------------
# Pure assessment: typed-distance median sets the principled window [median/f, median*f].
# typed_distances = [0.2, 0.3, 0.4] -> median 0.3; with factor 4 -> window [0.075, 1.2].
# ---------------------------------------------------------------------------
TYPED = np.array([0.2, 0.3, 0.4])


def test_assess_tau_principled_when_on_scale() -> None:
    a = assess_tau(0.3, TYPED, factor=4.0)
    assert a.regime == "principled"
    assert a.is_principled
    assert a.median_typed_distance == pytest.approx(0.3)
    assert a.recommended_tau == pytest.approx(0.3)
    assert a.low == pytest.approx(0.075)
    assert a.high == pytest.approx(1.2)


def test_assess_tau_too_sharp_below_window() -> None:
    a = assess_tau(0.05, TYPED, factor=4.0)  # 0.05 < 0.075
    assert a.regime == "too_sharp"
    assert not a.is_principled


def test_assess_tau_too_flat_above_window() -> None:
    a = assess_tau(2.0, TYPED, factor=4.0)  # 2.0 > 1.2
    assert a.regime == "too_flat"


def test_assess_tau_boundaries_are_inclusive_principled() -> None:
    assert assess_tau(0.075, TYPED, factor=4.0).regime == "principled"
    assert assess_tau(1.2, TYPED, factor=4.0).regime == "principled"


def test_assess_tau_undetermined_without_typed_distances() -> None:
    a = assess_tau(0.2, np.array([]), factor=4.0)
    assert a.regime == "undetermined"
    assert a.n_typed == 0


def test_format_tau_warning_reports_scale_and_consequence() -> None:
    msg_sharp = format_tau_warning(assess_tau(0.05, TYPED, factor=4.0))
    assert "0.3" in msg_sharp  # the typed-distance median scale
    assert "0.05" in msg_sharp  # the offending tau
    assert "winner-take-all" in msg_sharp
    assert "warn_tau=False" in msg_sharp

    msg_flat = format_tau_warning(assess_tau(2.0, TYPED, factor=4.0))
    assert "RI" in msg_flat  # degenerates to count-based RI


# ---------------------------------------------------------------------------
# Integration: MaRI.compute warns (only) when tau is off the dataset's typed scale.
# Two clusters (label A near [1,0], B near [0,1]); confounder splits each cluster so
# every sample has typed (SO/OS) cross-confounder neighbours at distance ~O(0.1-1).
# ---------------------------------------------------------------------------
def _toy_dataset(jitter_scale: float = 0.15) -> tuple[np.ndarray, "object"]:
    import pandas as pd

    n_per = 8
    rng = np.random.default_rng(0)
    rows = []
    feats = []
    for i in range(n_per):
        label = "A" if i % 2 == 0 else "B"
        conf = "C1" if i < n_per // 2 else "C2"
        base = np.array([1.0, 0.0]) if label == "A" else np.array([0.0, 1.0])
        jitter = jitter_scale * rng.standard_normal(2)
        feats.append(base + jitter)
        rows.append(
            {
                "sample_id": f"s{i}",
                "image_path": f"/tmp/{i}.png",
                "label": label,
                "scanner_vendor": conf,
                "group_id": f"slide-{i}",
                "dataset": "toy",
            }
        )
    return np.asarray(feats, dtype=float), pd.DataFrame(rows)


def _mari_kwargs(features, manifest):
    return dict(
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        k_candidates=[3],
        evaluation_design="all",
    )


def test_mari_compute_warns_on_tiny_tau() -> None:
    features, manifest = _toy_dataset()
    with pytest.warns(RuntimeWarning, match="winner-take-all"):
        MaRI.compute(**_mari_kwargs(features, manifest), tau=1e-4, warn_tau=True)


def test_mari_compute_silent_when_warn_tau_false() -> None:
    features, manifest = _toy_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        MaRI.compute(**_mari_kwargs(features, manifest), tau=1e-4, warn_tau=False)


def test_mari_compute_no_tau_warning_for_principled_tau() -> None:
    features, manifest = _toy_dataset()
    rec = MaRI.recommend_tau(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k=3,
        evaluation_design="all",
    )
    assert rec > 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # tau exactly at the recommended scale must not trigger the tau check
        # (other RuntimeWarnings, e.g. undefined coverage, are unrelated and filtered out)
        warnings.filterwarnings("ignore", message=".*undefined coverage.*")
        MaRI.compute(**_mari_kwargs(features, manifest), tau=float(rec), warn_tau=True)


# ---------------------------------------------------------------------------
# Auto tau: omitting tau resolves it to the dataset's typed-neighbour scale.
# ---------------------------------------------------------------------------
def test_omitting_tau_resolves_it_to_the_recommendation_at_the_operating_k() -> None:
    features, manifest = _toy_dataset()
    result = MaRI.compute(**_mari_kwargs(features, manifest))  # tau omitted -> auto

    expected = MaRI.recommend_tau(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k=result.k,
        evaluation_design="all",
    )
    assert result.tau == pytest.approx(expected)


def test_auto_tau_scores_identically_to_pinning_that_same_tau() -> None:
    """Auto is exactly 'pin tau to the recommendation' -- not a different estimator."""
    features, manifest = _toy_dataset()
    auto = MaRI.compute(**_mari_kwargs(features, manifest))
    pinned = MaRI.compute(**_mari_kwargs(features, manifest), tau=auto.tau)

    assert auto.k == pinned.k
    assert auto.value == pytest.approx(pinned.value)


def test_pinned_tau_is_reported_back_on_the_result() -> None:
    features, manifest = _toy_dataset()
    result = MaRI.compute(**_mari_kwargs(features, manifest), tau=0.37, warn_tau=False)
    assert result.tau == pytest.approx(0.37)


def test_ri_carries_no_tau() -> None:
    from croma import RI

    features, manifest = _toy_dataset()
    result = RI.compute(
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        k_candidates=[3],
        evaluation_design="all",
    )
    assert np.isnan(result.tau)


def test_auto_tau_never_warns_off_scale() -> None:
    """The median typed distance is on-scale by construction, so warn_tau has nothing to say."""
    features, manifest = _toy_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*undefined coverage.*")
        MaRI.compute(**_mari_kwargs(features, manifest), warn_tau=True)


def test_auto_tau_tracks_the_embedding_scale_but_a_pinned_tau_cannot() -> None:
    """The reason auto is the default: the on-scale tau is a property of the embedding.

    A tight embedding's typed neighbours sit closer than a diffuse one's, so the tau that
    keeps ``exp(-d / tau)`` in the graded regime differs between them. One fixed tau across
    both models sharpens the margin for one and flattens it for the other.
    """
    tight_features, tight_manifest = _toy_dataset(jitter_scale=0.02)
    loose_features, loose_manifest = _toy_dataset(jitter_scale=0.30)

    tight_tau = MaRI.compute(**_mari_kwargs(tight_features, tight_manifest)).tau
    loose_tau = MaRI.compute(**_mari_kwargs(loose_features, loose_manifest)).tau

    assert tight_tau < loose_tau


def test_auto_tau_falls_back_when_no_typed_neighbour_exists() -> None:
    """Label and confounder perfectly confounded -> no SO/OS neighbour -> no distance scale.

    MaRI is undefined for every sample here, so the fallback tau cannot move the score; it
    only has to keep ``exp(-d / tau)`` well-formed. But it must say so out loud.
    """
    import pandas as pd

    from croma.metrics.mari import TAU_FALLBACK

    manifest = pd.DataFrame(
        {
            "sample_id": ["a1", "a2", "b1", "b2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V1", "V2", "V2"],
            "group_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )
    features = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])

    with pytest.warns(RuntimeWarning, match="tau cannot be put on a distance scale"):
        result = MaRI.compute(
            features=features,
            manifest=manifest,
            confounder_column="scanner_vendor",
            k_candidates=[1],
            evaluation_design="all",
        )
    assert result.tau == pytest.approx(TAU_FALLBACK)
    assert result.undefined_frac == pytest.approx(1.0)


def test_auto_tau_fallback_on_a_collapsed_embedding_names_the_collapse() -> None:
    """Typed neighbours at distance zero is a different fault from having none at all.

    A collapsed encoder puts every typed neighbour at distance ``0``, so the median typed
    distance is ``0`` and auto-tau has no scale to sit on -- the same fallback, a different
    cause. MaRI is *not* undefined here: ``exp(-0 / tau) = 1`` at any positive tau, so the
    score degenerates to the count-based RI and is reported. A warning that said the score
    was undefined would invite the reader to ignore a number they are about to read.
    """
    from croma.metrics.mari import TAU_FALLBACK

    features, manifest = constant_embedding()

    with pytest.warns(RuntimeWarning) as caught:
        result = MaRI.compute(
            features=features,
            manifest=manifest,
            confounder_column=CONFOUNDER_COLUMN,
            k_candidates=[PINNED_K],
            evaluation_design="all",
        )

    tau_warnings = [str(w.message) for w in caught if "tau" in str(w.message)]
    assert len(tau_warnings) == 1
    message = tau_warnings[0]
    # The two claims the shared fallback branch used to make, both false here.
    assert "no typed" not in message
    assert "undefined" not in message
    # What is true instead: typed neighbours exist at distance 0, and MaRI degrades to RI.
    assert "distance 0" in message
    assert "collapsed" in message
    assert "RI" in message

    assert result.tau == pytest.approx(TAU_FALLBACK)
    assert np.isfinite(result.value)
    assert result.undefined_frac == pytest.approx(0.0)


@pytest.mark.parametrize("bad_tau", [0.0, -1.0])
def test_non_positive_tau_is_rejected(bad_tau: float) -> None:
    features, manifest = _toy_dataset()
    with pytest.raises(ValueError, match="tau must be > 0"):
        MaRI.compute(**_mari_kwargs(features, manifest), tau=bad_tau)
    with pytest.raises(ValueError, match="tau must be > 0"):
        MaRI.compute_curve(
            features=features,
            manifest=manifest,
            confounder_column="scanner_vendor",
            k_values=[3],
            evaluation_design="all",
            tau=bad_tau,
        )


def test_compute_curve_auto_tau_matches_the_curve_at_the_resolved_tau() -> None:
    features, manifest = _toy_dataset()
    resolved = MaRI.compute(**_mari_kwargs(features, manifest)).tau

    kwargs = dict(
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        k_values=[3, 5],
        evaluation_design="all",
    )
    auto_curve = MaRI.compute_curve(**kwargs)
    pinned_curve = MaRI.compute_curve(**kwargs, tau=resolved)

    assert auto_curve.keys() == pinned_curve.keys()
    for k in auto_curve:
        assert auto_curve[k] == pytest.approx(pinned_curve[k])


def test_operating_k_is_the_same_for_ri_and_mari() -> None:
    """k comes from kNN balanced accuracy, which never consults the weighting.

    This is what makes auto tau a two-step resolution rather than a circular one.
    """
    from croma import RI

    features, manifest = _toy_dataset()
    shared = dict(
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        k_candidates=[3, 5],
        evaluation_design="all",
    )
    assert RI.compute(**shared).k == MaRI.compute(**shared).k
