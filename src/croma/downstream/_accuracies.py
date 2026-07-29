"""The input shape downstream reductions share: a probe sweep's accuracy matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def as_accuracy_matrix(accuracies: ArrayLike) -> np.ndarray:
    """Coerce a confounder-biased probe sweep to a float matrix, or explain what is wrong.

    ``apd`` and ``nipd`` differ in their normalizer and reduction, but they
    read the same object: an ``(n_splits, n_iterations)`` matrix of balanced accuracies
    whose row ``0`` is the balanced baseline. That shared shape is checked once, here, so
    the reductions cannot drift into disagreeing about what a well-formed sweep is.

    Only conditions that hold for *both* reductions live here. Each reduction's own domain
    -- what its denominator needs in order to exist -- stays with the reduction, because
    the two denominators are different quantities and fail for different reasons.

    Args:
        accuracies: Anything array-like holding ``(n_splits, n_iterations)`` balanced
            accuracies, row ``0`` the balanced baseline.

    Returns:
        The same matrix as a 2-D ``float`` array.

    Raises:
        ValueError: If the input is not a 2-D matrix, holds no confounded split, holds no
            replicate, or holds a non-finite score.
    """
    acc = np.asarray(accuracies, dtype=float)
    if acc.ndim != 2:
        raise ValueError(
            f"accuracies must be a 2-D (n_splits, n_iterations) matrix, got {acc.ndim}-D"
        )
    if acc.shape[0] < 2:
        raise ValueError(
            "accuracies must hold the balanced baseline in row 0 and at least one "
            f"confounded split after it, got {acc.shape[0]} row(s)"
        )
    if acc.shape[1] < 1:
        raise ValueError("accuracies must hold at least one replicate per split, got 0")
    if not np.isfinite(acc).all():
        raise ValueError("accuracies must be finite")
    return acc
