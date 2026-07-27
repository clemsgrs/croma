"""nAPD: the skill-normalized Average Performance Drop of a confounder-biased probe."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def napd(accuracies: ArrayLike, chance: float) -> float:
    """Normalized Average Performance Drop: the share of *learnable* signal a
    confounder destroys.

    ``accuracies`` is an ``(n_splits, n_iterations)`` matrix of balanced-accuracy scores
    from a confounder-biased probe sweep, where row ``0`` is the balanced baseline and
    every later row is a progressively more confounded split. ``chance`` is the probe's
    chance level, ``1 / n_biological_classes`` -- exact, because the scorer is balanced
    accuracy. It is supplied by the caller rather than inferred, so the reduction also
    applies to sweeps this library did not produce.

    *Skill* is balanced accuracy above chance: the part of the score a confounder can
    actually destroy. nAPD is the mean skill ratio of the confounded splits against the
    baseline, minus one::

        skill_s = mean_i(accuracies[s][i]) - chance
        nAPD    = mean_{s>0}(skill_s / skill_0) - 1

    So ``0`` means no split lost skill, ``-1`` means the confounded splits fell all the
    way to chance, and a positive value means they beat the balanced baseline. This is
    APD's normalization changed from raw accuracy to skill, which makes the value
    comparable across tasks with different class counts.

    It deviates from PathoROB's ``compute_apd`` in exactly one way: it averages the
    ``n_iterations`` replicates *before* taking the ratio (ratio-of-means) rather than
    after (mean-of-ratios). The naive per-replicate form has ``skill_0`` of a *single*
    replicate in the denominator, so one replicate whose baseline dips toward chance
    dominates the average and can flip the sign. See
    ``docs/adr/0014-napd-averages-replicates-before-taking-the-ratio.md``.

    Args:
        accuracies: ``(n_splits, n_iterations)`` balanced accuracies, row ``0`` the
            balanced baseline. At least two splits and one replicate.
        chance: The probe's chance level, ``1 / n_biological_classes``, in ``[0, 1)``.

    Returns:
        The skill-normalized drop, a plain ``float``. Every admissible input yields a
        value: nAPD carries no skill floor and no undefined sentinel, because judging a
        cell too imprecise to interpret is a reporting decision, not a property of the
        metric.

    Raises:
        ValueError: If ``accuracies`` is not a 2-D matrix with at least one confounded
            split and at least one replicate, if it holds a non-finite score, if
            ``chance`` is outside ``[0, 1)``, or if the baseline's mean accuracy is at
            or below ``chance``. The last is a domain condition, not a skill floor: with
            no skill to lose there is no denominator, and below chance every ratio
            silently inverts its sign.
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
    if not 0.0 <= chance < 1.0:
        raise ValueError(f"chance must lie in [0, 1), got {chance}")

    skill = acc.mean(axis=1) - chance
    if skill[0] <= 0.0:
        raise ValueError(
            "the balanced baseline must score above chance for a skill ratio to exist, "
            f"got a mean baseline of {acc[0].mean()} against chance {chance}"
        )
    return float((skill[1:] / skill[0]).mean() - 1.0)
