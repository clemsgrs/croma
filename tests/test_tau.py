"""Tests for principled MaRI tau assessment and the warn_tau guidance in MaRI.compute."""

import warnings

import numpy as np
import pytest

from croma import MaRI
from croma.metrics.tau import TauAssessment, assess_tau, format_tau_warning


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
def _toy_dataset() -> tuple[np.ndarray, "object"]:
    import pandas as pd

    n_per = 8
    rng = np.random.default_rng(0)
    rows = []
    feats = []
    for i in range(n_per):
        label = "A" if i % 2 == 0 else "B"
        conf = "C1" if i < n_per // 2 else "C2"
        base = np.array([1.0, 0.0]) if label == "A" else np.array([0.0, 1.0])
        jitter = 0.15 * rng.standard_normal(2)
        feats.append(base + jitter)
        rows.append(
            {
                "sample_id": f"s{i}",
                "image_path": f"/tmp/{i}.png",
                "label": label,
                "scanner_vendor": conf,
                "slide_id": f"slide-{i}",
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
        evaluation_design="dataset_wide",
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
        evaluation_design="dataset_wide",
    )
    assert rec > 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # tau exactly at the recommended scale must not trigger the tau check
        # (other RuntimeWarnings, e.g. undefined coverage, are unrelated and filtered out)
        warnings.filterwarnings("ignore", message=".*undefined coverage.*")
        MaRI.compute(**_mari_kwargs(features, manifest), tau=float(rec), warn_tau=True)
