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
        confounder_column: str,
        k_candidates: list[int] | tuple[int, ...],
        evaluation_design: str = "paired_2x2",
        prune_ss_oo: bool = False,
        summarize_by_mean: bool = False,
    ) -> RobustnessResult:
        return cls._compute(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_candidates=k_candidates,
            evaluation_design=evaluation_design,
            prune_ss_oo=prune_ss_oo,
            summarize_by_mean=summarize_by_mean,
        )

    @classmethod
    def compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        confounder_column: str,
        k_values: list[int] | tuple[int, ...],
        evaluation_design: str = "paired_2x2",
        prune_ss_oo: bool = False,
        summarize_by_mean: bool = False,
    ) -> dict[int, float]:
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_values=k_values,
            evaluation_design=evaluation_design,
            prune_ss_oo=prune_ss_oo,
            summarize_by_mean=summarize_by_mean,
        )
