
import numpy as np
import pandas as pd

from mari.metrics.base import BaseRobustnessIndex
from mari.types import RobustnessResult


class MarginAwareRobustnessIndex(BaseRobustnessIndex):
    @classmethod
    def _weights(cls, distances: np.ndarray, **kwargs: float) -> np.ndarray:
        tau = float(kwargs["tau"])
        return np.exp(-distances / tau)

    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        mode: str,
        k_candidates: list[int] | tuple[int, ...],
        tau: float = 0.2,
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
    ) -> RobustnessResult:
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        return cls._compute(
            features=features,
            manifest=manifest,
            mode=mode,
            k_candidates=k_candidates,
            exclude_centers=exclude_centers,
            max_pairs=max_pairs,
            random_state=random_state,
            tau=float(tau),
        )

    @classmethod
    def compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        mode: str,
        k_values: list[int] | tuple[int, ...],
        tau: float = 0.2,
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
    ) -> dict[int, float]:
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            mode=mode,
            k_values=k_values,
            exclude_centers=exclude_centers,
            max_pairs=max_pairs,
            random_state=random_state,
            tau=float(tau),
        )
