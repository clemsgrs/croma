import warnings

import numpy as np
import pandas as pd

from croma.metrics.base import BaseRobustnessIndex
from croma.metrics.tau import assess_tau, format_tau_warning
from croma.types import RobustnessResult


class MarginAwareRobustnessIndex(BaseRobustnessIndex):
    @classmethod
    def _weights(cls, distances: np.ndarray, **kwargs: float) -> np.ndarray:
        tau = float(kwargs["tau"])
        return np.exp(-distances / tau)

    @classmethod
    def recommend_tau(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        confounder_column: str,
        k: int,
        evaluation_design: str = "paired_2x2",
    ) -> float:
        """Recommended ``tau`` for this dataset: the median typed (SO/OS) neighbour distance.

        Returns ``nan`` when no typed neighbour exists within the top-``k`` set (so ``tau``
        cannot be put on a meaningful scale).
        """
        typed = cls._collect_typed_neighbor_distances(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k=int(k),
            evaluation_design=evaluation_design,
        )
        return float(np.median(typed)) if typed.size else float("nan")

    @classmethod
    def _warn_if_tau_unprincipled(
        cls,
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        confounder_column: str,
        tau: float,
        k: int,
        evaluation_design: str,
    ) -> None:
        typed = cls._collect_typed_neighbor_distances(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k=int(k),
            evaluation_design=evaluation_design,
        )
        assessment = assess_tau(float(tau), typed)
        if assessment.regime in ("too_sharp", "too_flat"):
            warnings.warn(format_tau_warning(assessment), RuntimeWarning, stacklevel=2)

    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        confounder_column: str,
        k_candidates: list[int] | tuple[int, ...],
        tau: float = 0.2,
        evaluation_design: str = "paired_2x2",
        warn_tau: bool = True,
    ) -> RobustnessResult:
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        result = cls._compute(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_candidates=k_candidates,
            evaluation_design=evaluation_design,
            tau=float(tau),
        )
        if warn_tau:
            cls._warn_if_tau_unprincipled(
                features=features,
                manifest=manifest,
                confounder_column=confounder_column,
                tau=float(tau),
                k=int(result.k),
                evaluation_design=result.evaluation_design,
            )
        return result

    @classmethod
    def compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        confounder_column: str,
        k_values: list[int] | tuple[int, ...],
        tau: float = 0.2,
        evaluation_design: str = "paired_2x2",
    ) -> dict[int, float]:
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_values=k_values,
            evaluation_design=evaluation_design,
            tau=float(tau),
        )
