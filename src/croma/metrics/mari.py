import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from croma.metrics.base import BaseRobustnessIndex
from croma.metrics.tau import assess_tau, format_tau_warning
from croma.types import RobustnessResult

#: Temperature used only when auto-``tau`` cannot be put on a distance scale. Two different
#: datasets get here, and they are not the same situation:
#:
#: * no typed (SO/OS) neighbour exists within the top-``k`` set anywhere -- MaRI is undefined
#:   for every sample, so no ``tau`` changes anything;
#: * the median typed distance is exactly ``0`` -- a collapsed embedding (or duplicated rows).
#:   MaRI *is* defined here: ``exp(-0 / tau) = 1`` at any positive ``tau``, so the score
#:   degenerates to the count-based RI and again does not depend on this constant.
#:
#: Either way the choice cannot move the score; it only keeps ``exp(-d / tau)`` well-formed.
TAU_FALLBACK = 0.2


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

        Returns ``nan`` when no typed neighbour exists within the top-``k`` set, and ``0.0``
        when typed neighbours exist but half or more of them sit at distance ``0`` (a
        collapsed embedding, or a manifest that duplicates rows). Neither can be put on a
        meaningful scale, but they are distinct datasets: see :data:`TAU_FALLBACK`.
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
    def _resolve_tau(
        cls,
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        confounder_column: str,
        k: int,
        evaluation_design: str,
    ) -> float:
        """Auto-``tau`` at ``k``: the median typed-neighbour distance, or the fallback."""
        tau = cls.recommend_tau(
            features,
            manifest,
            confounder_column=confounder_column,
            k=int(k),
            evaluation_design=evaluation_design,
        )
        if not np.isfinite(tau):
            warnings.warn(
                f"no typed (SO/OS) neighbour within the top-{int(k)} set anywhere in the "
                f"dataset, so tau cannot be put on a distance scale; falling back to "
                f"tau={TAU_FALLBACK}. MaRI is undefined for every sample here, so this "
                f"choice does not affect the score.",
                RuntimeWarning,
                stacklevel=3,
            )
            return TAU_FALLBACK
        if tau <= 0.0:
            warnings.warn(
                f"the median typed (SO/OS) neighbour distance within the top-{int(k)} set "
                f"is 0, so tau cannot be put on a distance scale; falling back to "
                f"tau={TAU_FALLBACK}. Typed neighbours do exist here -- they sit at "
                f"distance 0, which is what a collapsed embedding, or a manifest that "
                f"duplicates rows, produces. The score is still defined and still reported: "
                f"exp(-0 / tau) = 1 at any positive tau, so on those neighbours MaRI carries "
                f"no margin information and degenerates to the count-based RI.",
                RuntimeWarning,
                stacklevel=3,
            )
            return TAU_FALLBACK
        return float(tau)

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
        tau: float | None = None,
        evaluation_design: str = "paired_2x2",
        warn_tau: bool = True,
    ) -> RobustnessResult:
        """Compute MaRI at the operating ``k`` selected by kNN balanced accuracy.

        Args:
            tau: Distance-decay temperature. Leave as ``None`` (recommended) to resolve it
                automatically as this dataset's median typed-neighbour distance at the
                operating ``k`` -- the on-scale value. A pinned ``tau`` is only comparable
                across models whose typed-neighbour distances share a scale; because that
                scale is a property of each embedding, a single fixed ``tau`` silently
                sharpens the margin for some models and flattens it for others.
            warn_tau: Warn when a pinned ``tau`` sits off the typed-neighbour scale. Ignored
                when ``tau`` is resolved automatically, which is on-scale by construction.
        """
        if tau is not None and float(tau) <= 0.0:
            raise ValueError("tau must be > 0")

        selected_k = cls._select_operating_k(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_candidates=k_candidates,
            evaluation_design=evaluation_design,
        )
        auto = tau is None
        resolved_tau = (
            cls._resolve_tau(
                features=features,
                manifest=manifest,
                confounder_column=confounder_column,
                k=selected_k,
                evaluation_design=evaluation_design,
            )
            if auto
            else float(tau)
        )

        artifacts = cls._compute_artifacts(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_values=k_candidates,
            evaluation_design=evaluation_design,
            selected_k=selected_k,
            include_selected_result=True,
            warn_selected_result=True,
            tau=resolved_tau,
        )
        if artifacts.result is None:
            raise RuntimeError("MaRI compute did not produce a selected-k result")
        result = replace(artifacts.result, tau=resolved_tau)

        if warn_tau and not auto:
            cls._warn_if_tau_unprincipled(
                features=features,
                manifest=manifest,
                confounder_column=confounder_column,
                tau=resolved_tau,
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
        tau: float | None = None,
        evaluation_design: str = "paired_2x2",
    ) -> dict[int, float]:
        """MaRI at every ``k`` in ``k_values``, all scored at a single ``tau``.

        Auto-``tau`` (the default) is resolved once, at the operating ``k`` selected over
        ``k_values``, and then held fixed across the sweep: a ``tau`` that moved with ``k``
        would confound the curve's shape with the temperature's.
        """
        if tau is not None and float(tau) <= 0.0:
            raise ValueError("tau must be > 0")
        if tau is None:
            selected_k = cls._select_operating_k(
                features=features,
                manifest=manifest,
                confounder_column=confounder_column,
                k_candidates=k_values,
                evaluation_design=evaluation_design,
            )
            tau = cls._resolve_tau(
                features=features,
                manifest=manifest,
                confounder_column=confounder_column,
                k=selected_k,
                evaluation_design=evaluation_design,
            )
        return cls._compute_curve(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            k_values=k_values,
            evaluation_design=evaluation_design,
            tau=float(tau),
        )
