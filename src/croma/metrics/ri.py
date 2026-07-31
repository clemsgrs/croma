import numpy as np
import pandas as pd

from croma.metrics.base import EVALUATION_DESIGN_ALL, BaseRobustnessIndex
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
        evaluation_design: str = EVALUATION_DESIGN_ALL,
    ) -> RobustnessResult:
        """Compute RI at the operating ``k`` selected by kNN balanced accuracy.

        Args:
            evaluation_design: ``"all"`` (the default) or ``"paired_2x2"``; see the
                constants in :mod:`croma.metrics.base` for what each scope scores.
        """
        return cls._compute(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_candidates=k_candidates,
            evaluation_design=evaluation_design,
        )

    @classmethod
    def compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        confounder_column: str,
        k_values: list[int] | tuple[int, ...],
        evaluation_design: str = EVALUATION_DESIGN_ALL,
    ) -> dict[int, float]:
        """RI at every ``k`` in ``k_values``, under ``evaluation_design`` (default ``"all"``)."""
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_values=k_values,
            evaluation_design=evaluation_design,
        )
