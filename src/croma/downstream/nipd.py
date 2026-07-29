"""nIPD: chance-normalized degradation integrated over confounder strength."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from croma.downstream._accuracies import as_accuracy_matrix


def nipd(accuracies: ArrayLike, cramers_v: ArrayLike, chance: float) -> float:
    """Normalized Integrated Performance Degradation over Cramér's V.

    ``accuracies`` holds one row per sampled Cramér's-V value and one column per
    repeated training run. Row ``0`` is the balanced baseline at ``V=0``. We first
    average balanced accuracy across repeats at each sampled value, then express the
    change from baseline as a fraction of baseline skill (accuracy above chance):

    .. code-block:: text

        g(V) = (mean_accuracy(V) - mean_accuracy(0)) / (mean_accuracy(0) - chance)

    nIPD is the signed area under ``g`` on ``[0, 1]``, estimated by trapezoidal
    integration over the supplied Cramér's-V coordinates. Thus ``0`` means no
    degradation, increasingly negative values mean greater shortcut susceptibility,
    and ``-0.5`` is the area of a linear fall from baseline performance at ``V=0`` to
    chance performance at ``V=1``.

    Args:
        accuracies: ``(n_splits, n_iterations)`` balanced accuracies, row ``0`` the
            balanced baseline.
        cramers_v: One finite, strictly increasing Cramér's-V value per accuracy row,
            beginning at ``0`` and ending at ``1``.
        chance: Balanced-accuracy chance level, ``1 / n_biological_classes``.

    Returns:
        The signed trapezoidal area under the normalized degradation curve.

    Raises:
        ValueError: If the accuracy matrix is malformed, the Cramér's-V coordinates
            are malformed or do not span ``[0, 1]``, chance lies outside ``[0, 1)``,
            or mean baseline balanced accuracy is at or below chance.
    """
    acc = as_accuracy_matrix(accuracies)
    v = np.asarray(cramers_v, dtype=float)
    if v.ndim != 1 or len(v) != len(acc):
        raise ValueError(
            "cramers_v must be a 1-D vector with one Cramér's-V value per accuracy row, "
            f"got shape {v.shape} for {len(acc)} rows"
        )
    if (
        not np.isfinite(v).all()
        or not np.all(np.diff(v) > 0.0)
        or not (np.isclose(v[0], 0.0) and np.isclose(v[-1], 1.0))
    ):
        raise ValueError("cramers_v must be finite, strictly increasing, and span [0, 1]")
    if not 0.0 <= chance < 1.0:
        raise ValueError(f"chance must lie in [0, 1), got {chance}")

    mean_accuracy = acc.mean(axis=1)
    baseline_skill = mean_accuracy[0] - chance
    if baseline_skill <= 0.0:
        raise ValueError(
            "the balanced baseline must score above chance for normalized degradation "
            f"to exist, got a mean baseline of {mean_accuracy[0]} against chance {chance}"
        )
    degradation = (mean_accuracy - mean_accuracy[0]) / baseline_skill
    return float(np.sum(np.diff(v) * (degradation[:-1] + degradation[1:]) / 2.0))
