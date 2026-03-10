
import numpy as np
import pandas as pd

from croma.metrics.base import BaseRobustnessIndex
from croma.types import RobustnessResult


class RobustnessIndex(BaseRobustnessIndex):
    @classmethod
    def _weights(cls, distances: np.ndarray, **kwargs: float) -> np.ndarray:
        return np.ones_like(distances, dtype=float)

    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        k_candidates: list[int] | tuple[int, ...],
        evaluation_design: str = "paired_2x2",
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
    ) -> RobustnessResult:
        return cls._compute(
            features=features,
            manifest=manifest,
            k_candidates=k_candidates,
            evaluation_design=evaluation_design,
            exclude_centers=exclude_centers,
            max_pairs=max_pairs,
            random_state=random_state,
        )

    @classmethod
    def compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        k_values: list[int] | tuple[int, ...],
        evaluation_design: str = "paired_2x2",
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
    ) -> dict[int, float]:
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            k_values=k_values,
            evaluation_design=evaluation_design,
            exclude_centers=exclude_centers,
            max_pairs=max_pairs,
            random_state=random_state,
        )
