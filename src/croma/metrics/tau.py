"""Principled selection of the MaRI temperature ``tau``.

MaRI weights neighbour evidence by ``exp(-d / tau)``, where ``d`` is the cosine distance to
a typed (``SO``/``OS``) neighbour. The weights are not normalised per anchor: pooled MaRI is a
ratio of sums over every typed slot of every anchor. So ``tau`` acts at two levels.

* **Within one anchor**, only distance *differences* matter. Relative to the anchor's nearest
  neighbour at ``d_1``, a neighbour at ``d`` receives ``exp(-(d - d_1) / tau)``. Inside a
  fixed-``k`` neighbourhood the typed neighbours lie in a narrow distance band, much narrower
  than the distances themselves. On the scale of the typed distances the weights inside one
  neighbourhood are therefore close to flat, and even a few times sharper they are only
  moderately graded. Reading ``exp(-d / tau)`` as if the nearest neighbour sat at distance
  zero overstates this grading badly.
* **Across anchors**, absolute distances matter. An anchor whose typed neighbours sit closer
  contributes more weight to the pooled sums, and this comparison between *different* anchors
  carries most of the difference between MaRI and the count-based RI.

The useful range of ``tau`` is still fixed by the *scale* of the typed-neighbour distances:

* ``tau`` much smaller than the typical typed distance -> the pooled score is carried by the
  few anchors whose typed neighbours are closest, and becomes noisy.
* ``tau`` much larger -> ``exp(-d / tau) ~ 1`` for every neighbour, so MaRI degenerates into
  the count-based RI and the margin information is lost.

We recommend the median typed-neighbour distance, which the metric resolves automatically.
Because it scales with the distances, MaRI is unchanged when every distance is multiplied by a
common factor, so encoders whose distance scales differ are weighted alike; a pinned ``tau``
lacks this property. Sharper temperatures move MaRI further from RI, but on the benchmarks
studied they do not consistently strengthen its association with downstream shortcut
susceptibility, and on some they weaken it, so there is no case for sharpening below the
median. ``assess_tau`` flags a ``tau`` more than ``factor``
away from the median in either direction (default ``factor=4``) as off the distance scale.
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
