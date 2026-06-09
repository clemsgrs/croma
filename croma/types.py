from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustnessResult:
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
    undefined_frac: float = 0.0
    ss_dominated_undefined_frac: float = 0.0
    oo_dominated_undefined_frac: float = 0.0
    mixed_undefined_frac: float = 0.0
    evaluation_design: str = "paired_2x2"
    evaluation_unit: str = "occurrence"
    alpha: float = 0.10
    median_value: float = float("nan")
    q_alpha: float = float("nan")
    ltm_alpha: float = float("nan")


@dataclass(frozen=True)
class CCMRResult:
    dataset: str
    m: int
    value: float
    std: float
    n_pairs: int
    pair_values: np.ndarray
    sample_values: np.ndarray
    sample_values_aligned: np.ndarray
    occurrence_defined_mask: np.ndarray
    undefined_frac: float
    evaluation_design: str = "paired_2x2"
    evaluation_unit: str = "occurrence"
    occurrence_subsets: np.ndarray | None = None
    occurrence_source_indices: np.ndarray | None = None
    k_start: int = 0
    k_final: int = 0
    retries: int = 0
    alpha: float = 0.10
    q_alpha: float = float("nan")
    ltm_alpha: float = float("nan")
