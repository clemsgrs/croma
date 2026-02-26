from __future__ import annotations

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


@dataclass(frozen=True)
class CCRRResult:
    dataset: str
    m: int
    value: float
    std: float
    n_pairs: int
    pair_values: np.ndarray
    sample_values: np.ndarray
    undefined_frac: float
    acceptance_threshold: float = 0.0
    acceptance_met: bool = True
    k_start: int = 0
    k_final: int = 0
    retries: int = 0
    alpha: float = 0.10
    q_alpha: float = float("nan")
    ltm_alpha: float = float("nan")
