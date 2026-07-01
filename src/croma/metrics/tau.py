"""Principled selection of the MaRI temperature ``tau``.

MaRI weights neighbour evidence by ``exp(-d / tau)``, where ``d`` is the cosine distance to
a typed (``SO``/``OS``) neighbour. The useful range of ``tau`` is fixed by the *scale* of
those typed-neighbour distances on the dataset at hand:

* ``tau`` much smaller than the typical typed distance -> ``exp(-d / tau)`` collapses onto
  the single nearest typed neighbour (winner-take-all); the score becomes noisy and
  dominated by one sample.
* ``tau`` much larger -> ``exp(-d / tau) ~ 1`` for every neighbour, so MaRI degenerates into
  the count-based RI and the margin information is lost.

A principled ``tau`` therefore lies *on the scale* of the typed-neighbour distances. We take
the median typed-neighbour distance as the recommended value and flag ``tau`` that falls more
than ``factor`` away from it in either direction. The default ``factor=4`` brackets the
"graded" regime: at ``tau = median / 4`` the median typed neighbour already receives only
``exp(-4) ~ 0.018`` of the nearest neighbour's weight (effectively winner-take-all), while at
``tau = median * 4`` the weights across a typical typed shell are nearly uniform (~RI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TAU_SCALE_FACTOR = 4.0


@dataclass(frozen=True)
class TauAssessment:
    """Where a chosen ``tau`` sits relative to a dataset's typed-neighbour distance scale."""

    tau: float
    n_typed: int
    median_typed_distance: float
    recommended_tau: float
    low: float
    high: float
    factor: float
    regime: str  # "principled" | "too_sharp" | "too_flat" | "undetermined"

    @property
    def is_principled(self) -> bool:
        return self.regime == "principled"


def assess_tau(
    tau: float,
    typed_distances: np.ndarray,
    *,
    factor: float = DEFAULT_TAU_SCALE_FACTOR,
) -> TauAssessment:
    """Classify ``tau`` against the median typed-neighbour distance.

    ``typed_distances`` are the cosine distances to the ``SO``/``OS`` neighbours actually
    used by MaRI. The principled window is ``[median / factor, median * factor]``.
    """
    tau = float(tau)
    factor = float(factor)
    d = np.asarray(typed_distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        nan = float("nan")
        return TauAssessment(tau, 0, nan, nan, nan, nan, factor, "undetermined")

    median = float(np.median(d))
    low = median / factor
    high = median * factor
    if tau < low:
        regime = "too_sharp"
    elif tau > high:
        regime = "too_flat"
    else:
        regime = "principled"
    return TauAssessment(tau, int(d.size), median, median, low, high, factor, regime)


def format_tau_warning(assessment: TauAssessment) -> str:
    """A verbose, actionable message explaining why ``tau`` is off-scale and what to use."""
    a = assessment
    if a.regime == "too_sharp":
        consequence = (
            f"tau={a.tau:.3g} is well below this scale (< median/{a.factor:g} = {a.low:.3g}): "
            "exp(-d/tau) collapses onto the single nearest typed neighbour (winner-take-all), "
            "so MaRI becomes noisy and dominated by one sample per neighbourhood."
        )
    elif a.regime == "too_flat":
        consequence = (
            f"tau={a.tau:.3g} is well above this scale (> median*{a.factor:g} = {a.high:.3g}): "
            "exp(-d/tau) ~ 1 for every neighbour, so MaRI degenerates into the count-based RI "
            "and the margin information is lost."
        )
    else:
        consequence = ""
    return (
        "MaRI tau may not be well matched to this dataset. "
        f"The typed (SO/OS) neighbour cosine distances have median "
        f"{a.median_typed_distance:.3g} (n={a.n_typed}); a principled tau lies on that scale, "
        f"i.e. roughly [{a.low:.3g}, {a.high:.3g}] (recommended ~ {a.recommended_tau:.3g}). "
        f"{consequence} "
        "Set tau closer to the recommended value, or silence this check with warn_tau=False."
    ).replace("  ", " ")
