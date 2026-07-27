"""APD: PathoROB's Average Performance Drop, the faithful reference reduction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from croma.downstream._accuracies import as_accuracy_matrix
from croma.downstream._pathorob import compute_apd as _pathorob_compute_apd


def apd(accuracies: ArrayLike) -> float:
    """Average Performance Drop: the mean relative accuracy change a confounder costs.

    ``accuracies`` is an ``(n_splits, n_iterations)`` matrix of balanced-accuracy scores
    from a confounder-biased probe sweep, where row ``0`` is the balanced baseline and
    every later row is a progressively more confounded split. APD is the mean, over the
    confounded splits, of each split's accuracy relative to the baseline, minus one::

        APD = mean_i( mean_{s>0}( accuracies[s][i] / accuracies[0][i] ) - 1 )

    So ``0`` means no split lost accuracy, and a value near ``-1`` that the confounded
    splits collapsed. Closer to zero is more robust.

    This is **PathoROB's** metric, not croma's, and it is reported because it is
    faithful: the reduction itself is vendored verbatim from PathoROB
    (``croma.downstream._pathorob``) rather than re-derived, so a number reported as
    "APD" means what the PathoROB paper means by it. This function adds argument
    validation and averages the replicate axis -- exactly what PathoROB's own driver
    does to turn the per-replicate scores into the scalar it publishes.

    Note the ratio is taken **per replicate** and averaged afterwards (mean-of-ratios).
    :func:`croma.napd` averages the replicates first (ratio-of-means), and additionally
    divides by skill rather than raw accuracy. The two differences are deliberate and
    are not aligned: see
    ``docs/adr/0014-napd-averages-replicates-before-taking-the-ratio.md``.

    Args:
        accuracies: ``(n_splits, n_iterations)`` balanced accuracies, row ``0`` the
            balanced baseline. At least two splits and one replicate.

    Returns:
        The mean relative performance drop, a plain ``float``.

    Raises:
        ValueError: If ``accuracies`` is not a 2-D matrix with at least one confounded
            split and at least one replicate, if it holds a non-finite score, or if any
            replicate's baseline accuracy is zero or negative. The last is the
            reduction's domain running out: mean-of-ratios divides by each replicate's
            own baseline, so a zero there has no ratio and a negative one inverts every
            sign.
    """
    acc = as_accuracy_matrix(accuracies)
    if not (acc[0] > 0.0).all():
        raise ValueError(
            "every replicate's baseline accuracy must be positive for a ratio to exist, "
            f"got a minimum baseline of {acc[0].min()}"
        )

    return float(np.mean(_pathorob_compute_apd(acc)))
