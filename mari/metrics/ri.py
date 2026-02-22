from __future__ import annotations

import numpy as np
import pandas as pd

from mari.metrics.base import BaseRobustnessIndex
from mari.types import RobustnessResult


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
        mode: str,
        k_candidates: list[int] | tuple[int, ...],
        max_pairs: int | None = None,
        random_state: int = 0,
    ) -> RobustnessResult:
        return cls._compute(
            features=features,
            manifest=manifest,
            mode=mode,
            k_candidates=k_candidates,
            max_pairs=max_pairs,
            random_state=random_state,
        )
