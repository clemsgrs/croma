"""MaRI's two tau limits, and the separation from RI that the package exists to provide.

``croma.metrics.tau`` justifies a user-facing runtime warning with two limiting claims: as
``tau`` grows the weights ``exp(-d / tau)`` approach 1 and MaRI degenerates into the
count-based RI, and as ``tau`` shrinks they collapse onto the single nearest typed neighbour.
Both are tested here as equalities against RI and against ``{0, 1}``, not as the inequality
``sharp > flat`` that any monotone function would satisfy.

**The sharp limit is reachable, but it has a floor.** For the count-matched pair, whose typed
neighbours sit at exactly 0.05 and 0.30, winner-take-all is exact for every ``tau`` below
``6.87e-3`` -- and it survives only down to ``4.03e-4``, where ``exp(-0.30 / tau)`` underflows
to zero and the samples whose only typed evidence is a far neighbour go undefined; below
``6.71e-5`` even the near neighbour underflows and *every* sample is undefined, so MaRI
reports total undefined coverage and falls back to its neutral 0.5 rather than to a
single-neighbour vote. Both ends of that window are asserted below. The docstring's claim
therefore holds over four decades of ``tau`` and fails only beneath the float64 underflow
floor, which is why ``tau.py`` is left as it stands.
"""

import warnings

import numpy as np
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
#: nearest one's weight, flat enough that no weight has underflowed to zero yet.
SHARP_TAU = 1e-3

#: A ``tau`` beneath the pair's underflow floor, where every ``exp(-d / tau)`` is exactly 0.
UNDERFLOW_TAU = 1e-6

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
# The sharp limit: tau -> 0 collapses onto the single nearest typed neighbour, until the
# weights underflow.
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
            evaluation_design="dataset_wide",
            tau=SHARP_TAU,
            warn_tau=False,
        )

    assert result.undefined_frac == 0.0
    assert result.sample_values == pytest.approx(expected, abs=WINNER_TAKE_ALL_TOLERANCE)


@pytest.mark.parametrize("case", sorted(SHARP_CASES), ids=sorted(SHARP_CASES))
def test_sharp_tau_below_the_underflow_floor_leaves_mari_undefined(case: str) -> None:
    """The floor of the sharp limit: past it MaRI is undefined, not winner-take-all.

    ``exp(-d / tau)`` underflows to exactly 0 for every neighbour, so both sides of the
    SO/OS ratio vanish and no sample carries typed evidence any more. This is the boundary of
    the regime the test above confirms -- and MaRI does say so, via its undefined coverage,
    rather than silently returning a single-neighbour vote.
    """
    features, manifest, _expected = SHARP_CASES[case]
    with pytest.warns(RuntimeWarning, match="undefined coverage"):
        result = MaRI.compute(
            features,
            manifest,
            confounder_column="scanner_vendor",
            k_candidates=[PINNED_K],
            evaluation_design="dataset_wide",
            tau=UNDERFLOW_TAU,
            warn_tau=False,
        )

    assert result.undefined_frac == pytest.approx(1.0)
    assert result.sample_values.size == 0


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
