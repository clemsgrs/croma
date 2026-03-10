
import numpy as np
import pandas as pd

from croma.metrics.base import BaseRobustnessIndex
from croma.types import RobustnessResult


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
        k_candidates: list[int] | tuple[int, ...],
        tau: float = 0.2,
        evaluation_design: str = "paired_2x2",
    ) -> RobustnessResult:
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        return cls._compute(
            features=features,
            manifest=manifest,
            k_candidates=k_candidates,
            evaluation_design=evaluation_design,
            tau=float(tau),
        )

    @classmethod
    def compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        k_values: list[int] | tuple[int, ...],
        tau: float = 0.2,
        evaluation_design: str = "paired_2x2",
    ) -> dict[int, float]:
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            k_values=k_values,
            evaluation_design=evaluation_design,
            tau=float(tau),
        )
