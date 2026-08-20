"""MaRI's two tau limits, and the separation from RI that the package exists to provide.

``croma.metrics.tau`` justifies a user-facing runtime warning with two limiting claims: as
``tau`` grows the weights ``exp(-d / tau)`` approach 1 and MaRI degenerates into the
count-based RI, and as ``tau`` shrinks they collapse onto the single nearest typed neighbour.
Both are tested here as equalities against RI and against ``{0, 1}``, not as the inequality
``sharp > flat`` that any monotone function would satisfy.

**The sharp limit has no numerical support floor.** For the count-matched pair, whose typed
neighbours sit at exactly 0.05 and 0.30, winner-take-all is exact for every ``tau`` below
``6.87e-3``. At extreme positive temperatures the raw exponentials underflow, but the stable
log-domain ratio preserves both that mathematical limit and the same defined anchors as RI.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from metric_harness import (
    COUNT_MATCHED_RI,
    NAMED_EMBEDDINGS,
    PINNED_K,
    compute_metric,
    count_matched_pair,
)

from croma import RI, MaRI

#: A ``tau`` far above every typed-neighbour distance in these fixtures, where
#: ``exp(-d / tau) = 1 - d / tau`` differs from 1 by ~1e-12. Weights are then uniform to well
#: within :data:`FLAT_TOLERANCE`, so MaRI's weighted ratio is RI's count ratio.
FLAT_TAU = 1e12
FLAT_TOLERANCE = 1e-9

#: A ``tau`` inside the count-matched pair's winner-take-all window (see the module
#: docstring): sharp enough that the second-nearest typed neighbour contributes ~1e-109 of the
#: nearest one's weight.
SHARP_TAU = 1e-3

#: The smallest positive float64 ``tau``: raw weights underflow and even ``d / tau``
#: overflows in float64, exercising the full valid positive-temperature range.
UNDERFLOW_TAU = float(np.nextafter(0.0, 1.0))

#: Winner-take-all is attained exactly from above (the sum snaps to the nearest weight) but
#: only approached from below, where the ratio bottoms out near 1e-109 rather than at 0.
WINNER_TAKE_ALL_TOLERANCE = 1e-12

#: How far apart MaRI must place the two count-matched embeddings. They score ~0.997 and
#: ~0.465 against a shared RI of 2/3, so the realised gap is ~0.53.
MARI_SEPARATION = 0.5


def _named_cases() -> dict[str, tuple[np.ndarray, "object"]]:
    """Every named embedding, plus both halves of the count-matched pair."""
    pair = count_matched_pair()
    cases = {name: ctor() for name, ctor in NAMED_EMBEDDINGS.items()}
    cases["count_matched_so_near"] = (pair.so_near, pair.manifest)
    cases["count_matched_os_near"] = (pair.os_near, pair.manifest)
    return cases


FLAT_CASES = _named_cases()


# ---------------------------------------------------------------------------
# The flat limit: tau -> inf turns MaRI back into RI.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(FLAT_CASES), ids=sorted(FLAT_CASES))
def test_flat_tau_makes_mari_agree_with_ri(case: str) -> None:
    """At a large pinned tau, MaRI and RI agree on the same fixture at the same k.

    ``warn_tau=False`` on purpose: the point of the test is to enter the very regime the
    warning exists to keep users out of.
    """
    features, manifest = FLAT_CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ri = compute_metric(RI, features, manifest, k=PINNED_K)
        mari = compute_metric(MaRI, features, manifest, k=PINNED_K, tau=FLAT_TAU, warn_tau=False)

    assert mari.score == pytest.approx(ri.score, abs=FLAT_TOLERANCE)
    assert mari.sample_values == pytest.approx(ri.sample_values, abs=FLAT_TOLERANCE)


def test_the_flat_limit_is_not_vacuous() -> None:
    """MaRI is not RI in disguise: on the same fixture, at its own tau, it disagrees.

    Without this the flat-limit test above would pass for a MaRI that ignored distance
    entirely.
    """
    pair = count_matched_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ri = compute_metric(RI, pair.so_near, pair.manifest, k=PINNED_K)
        mari = compute_metric(MaRI, pair.so_near, pair.manifest, k=PINNED_K)

    assert abs(mari.score - ri.score) > 0.3


# ---------------------------------------------------------------------------
# The sharp limit: tau -> 0 collapses onto the single nearest typed neighbour.
# ---------------------------------------------------------------------------
def _sharp_limit_cases() -> dict[str, tuple[np.ndarray, "object", np.ndarray]]:
    """The pair, each half with the winner-take-all value its geometry dictates.

    Per block the four slots are (centre, SO partner, SO partner, OS partner). In
    ``so_near`` the centre's nearest typed neighbour is one of its two SO partners, so it goes
    to 1; in ``os_near`` it is the OS partner, so it goes to 0. The three non-centre slots
    have a single typed neighbour each -- the centre -- which is trivially their nearest: SO
    for the two SO partners (1), OS for the OS partner (0).
    """
    pair = count_matched_pair()
    return {
        "so_near": (pair.so_near, pair.manifest, np.tile([1.0, 1.0, 1.0, 0.0], pair.n_blocks)),
        "os_near": (pair.os_near, pair.manifest, np.tile([0.0, 1.0, 1.0, 0.0], pair.n_blocks)),
    }


SHARP_CASES = _sharp_limit_cases()


@pytest.mark.parametrize("case", sorted(SHARP_CASES), ids=sorted(SHARP_CASES))
def test_sharp_tau_collapses_onto_the_nearest_typed_neighbour(case: str) -> None:
    """Inside the winner-take-all window every sample is 1 (nearest typed is SO) or 0 (OS)."""
    features, manifest, expected = SHARP_CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = MaRI.compute(
            features,
            manifest,
            confounder_column="scanner_vendor",
            k_candidates=[PINNED_K],
            evaluation_design="all",
            tau=SHARP_TAU,
            warn_tau=False,
        )

    assert result.undefined_frac == 0.0
    assert result.sample_values == pytest.approx(expected, abs=WINNER_TAKE_ALL_TOLERANCE)


@pytest.mark.parametrize("evaluation_design", ["all", "paired_2x2"])
def test_extreme_tau_preserves_mari_ratio_and_support(evaluation_design: str) -> None:
    """Distance weighting never changes support, even when raw weights underflow."""
    features, manifest, expected_values = SHARP_CASES["so_near"]
    if evaluation_design == "paired_2x2":
        manifest = manifest.assign(subset="pair")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shared = {
            "features": features,
            "manifest": manifest,
            "confounder_column": "scanner_vendor",
            "k_candidates": [PINNED_K],
            "evaluation_design": evaluation_design,
        }
        ri = RI.compute(**shared)
        mari = MaRI.compute(
            **shared,
            tau=UNDERFLOW_TAU,
            warn_tau=False,
        )

    expected_support = np.ones(len(manifest), dtype=bool)
    np.testing.assert_array_equal(ri.occurrence_defined_mask, expected_support)
    np.testing.assert_array_equal(mari.occurrence_defined_mask, expected_support)
    np.testing.assert_array_equal(mari.sample_values_aligned, expected_values)
    assert ri.undefined_frac == 0.0
    assert mari.undefined_frac == 0.0
    assert mari.value == 1.0


def test_subnormal_weights_preserve_the_mathematical_mari_ratio() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [0.256, np.sqrt(1.0 - 0.256**2)],  # SO distance from query: 0.744
            [0.255, -np.sqrt(1.0 - 0.255**2)],  # OS distance from query: 0.745
            [0.0, 1.0],
        ],
        dtype=float,
    )
    manifest = pd.DataFrame(
        {
            "sample_id": ["query", "so", "os", "far"],
            "image_path": ["/query", "/so", "/os", "/far"],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2"],
            "group_id": ["g0", "g1", "g2", "g3"],
            "dataset": ["toy"] * 4,
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = MaRI.compute(
            features,
            manifest,
            confounder_column="scanner_vendor",
            k_candidates=[2],
            tau=0.001,
            warn_tau=False,
        )

    assert result.sample_values_aligned[0] == pytest.approx(0.7310585786300049, abs=1e-15)


# ---------------------------------------------------------------------------
# Same RI, different MaRI -- the package's thesis, as an assertion.
# ---------------------------------------------------------------------------
def test_the_count_matched_pair_is_invisible_to_ri() -> None:
    """Identical neighbour counts at the pinned k, so RI cannot tell the two models apart.

    Not RI's tie point either: the pair sits at 2/3, so this is not a matching engineered by
    handing every sample one SO and one OS.
    """
    pair = count_matched_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        so_near = compute_metric(RI, pair.so_near, pair.manifest, k=PINNED_K)
        os_near = compute_metric(RI, pair.os_near, pair.manifest, k=PINNED_K)

    assert so_near.score == pytest.approx(COUNT_MATCHED_RI, abs=1e-12)
    assert os_near.score == pytest.approx(so_near.score, abs=1e-12)
    assert so_near.sample_values == pytest.approx(os_near.sample_values, abs=1e-12)


def test_mari_separates_what_ri_cannot() -> None:
    """The same two models, told apart by their neighbourhood margins.

    ``so_near`` keeps its SO neighbours close and pushes its OS neighbour out; ``os_near``
    does the reverse. MaRI lands them on opposite sides of the RI they share.
    """
    pair = count_matched_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        so_near = compute_metric(MaRI, pair.so_near, pair.manifest, k=PINNED_K)
        os_near = compute_metric(MaRI, pair.os_near, pair.manifest, k=PINNED_K)

    assert so_near.score - os_near.score > MARI_SEPARATION
    assert so_near.score > COUNT_MATCHED_RI > os_near.score
