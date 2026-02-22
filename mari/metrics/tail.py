from __future__ import annotations

import numpy as np


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not (0.0 <= a <= 100.0):
        raise ValueError("alpha must be in [0, 100]")
    return a


def _validate_values(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be a 1-D array")
    if arr.size <= 0:
        raise ValueError("values must not be empty")
    return arr


def tail_percentile(values: np.ndarray | list[float], alpha: float) -> float:
    arr = _validate_values(values)
    a = _validate_alpha(alpha)
    return float(np.percentile(arr, a))


def lower_tail_mean(values: np.ndarray | list[float], alpha: float) -> float:
    arr = _validate_values(values)
    q = tail_percentile(arr, alpha)
    tail = arr[arr <= q]
    if tail.size <= 0:
        raise RuntimeError("lower-tail set is empty; cannot compute mean")
    return float(tail.mean())

