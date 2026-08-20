from dataclasses import KW_ONLY, dataclass

import numpy as np


@dataclass(frozen=True)
class RobustnessResult:
    """One RI or MaRI evaluation and the units supporting its pooled score.

    ``support`` is the canonical aggregate: the fraction of sample or occurrence units
    with defined RI/MaRI values. The cause-specific undefined fractions retain all
    evaluation units as their denominator.
    """

    dataset: str
    k: int
    value: float
    std: float
    n_pairs: int
    pair_values: np.ndarray
    sample_values: np.ndarray
    sample_values_aligned: np.ndarray
    occurrence_defined_mask: np.ndarray
    sample_undefined_types: np.ndarray
    occurrence_subsets: np.ndarray
    occurrence_source_indices: np.ndarray
    _: KW_ONLY
    support: float
    ss_dominated_undefined_frac: float = 0.0
    oo_dominated_undefined_frac: float = 0.0
    mixed_undefined_frac: float = 0.0
    evaluation_design: str = "all"
    evaluation_unit: str = "sample"
    alpha: float = 0.10
    median_value: float = float("nan")
    q_alpha: float = float("nan")
    ltm_alpha: float = float("nan")
    # The temperature MaRI actually scored with, whether the caller pinned it or it was
    # resolved automatically. ``nan`` on RI, which carries no temperature.
    tau: float = float("nan")


@dataclass(frozen=True)
class CRoMaResult:
    """One CRoMa evaluation: the pooled margin, its distribution, and its tail.

    ``f0`` is the confounder-dominant fraction :math:`F(0)` -- the empirical CDF of the
    per-sample margin at zero, i.e. the fraction of evaluation units whose margin is
    ``<= 0``. Exact zero counts as confounder-dominant. CRoMa requires total support, so a
    result exists only when every requested sample or subset occurrence is scoreable.
    """

    dataset: str
    m: int
    value: float
    std: float
    n_pairs: int
    pair_values: np.ndarray
    sample_values: np.ndarray
    sample_values_aligned: np.ndarray
    _: KW_ONLY
    evaluation_design: str = "all"
    evaluation_unit: str = "sample"
    occurrence_subsets: np.ndarray | None = None
    occurrence_source_indices: np.ndarray | None = None
    k_start: int = 0
    k_final: int = 0
    retries: int = 0
    alpha: float = 0.10
    q_alpha: float = float("nan")
    ltm_alpha: float = float("nan")
    f0: float = float("nan")
