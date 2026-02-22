from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from mari.metrics.neighbors import _optimal_k_by_knn_balanced_accuracy, _prepare_neighbors
from mari.metrics.pairs import ensure_required_columns, infer_2x2_pairs, subset_by_pair
from mari.types import RobustnessResult


def _ratio_or_default(so: float, os: float, default: float = 0.5) -> float:
    denom = float(so + os)
    if denom <= 0:
        return float(default)
    return float(float(so) / denom)


class BaseRobustnessIndex(ABC):
    _PAIR_MODES = {"paired", "global"}

    @classmethod
    def _infer_dataset_name(cls, df: pd.DataFrame) -> str:
        if "dataset" not in df.columns or len(df) == 0:
            return "dataset"
        return str(df["dataset"].iloc[0]).strip() or "dataset"

    @classmethod
    def _validate_inputs(cls, features: np.ndarray, manifest: pd.DataFrame) -> None:
        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas.DataFrame")
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a numpy.ndarray")
        if features.ndim != 2:
            raise ValueError("features must be a 2-D array of shape (N, D)")
        if len(features) != len(manifest):
            raise ValueError("features row count must match manifest row count")
        ensure_required_columns(manifest, "manifest")

    @classmethod
    @abstractmethod
    def _weights(cls, distances: np.ndarray, **kwargs: float) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _score_from_neighbors(
        cls,
        labels: np.ndarray,
        centers: np.ndarray,
        neigh_idx: np.ndarray,
        neigh_dist: np.ndarray,
        valid_counts: np.ndarray,
        k: int,
        **kwargs: float,
    ) -> tuple[float, np.ndarray]:
        target_k = int(k)
        so_total = 0.0
        os_total = 0.0
        sample_scores = np.full((len(labels),), 0.5, dtype=float)

        for i in range(len(labels)):
            eff_k = min(target_k, int(valid_counts[i]))
            if eff_k <= 0:
                continue

            row_idx = neigh_idx[i, :eff_k]
            row_dist = neigh_dist[i, :eff_k]
            keep = row_idx >= 0
            if not bool(np.any(keep)):
                continue

            row_idx = row_idx[keep]
            row_dist = row_dist[keep]

            neigh_labels = labels[row_idx]
            neigh_centers = centers[row_idx]
            sample_label = labels[i]
            sample_center = centers[i]

            weights = cls._weights(row_dist, **kwargs)
            if weights.shape != row_dist.shape:
                raise ValueError("weight function must return one weight per neighbor distance")

            so_mask = np.logical_and(neigh_labels == sample_label, neigh_centers != sample_center)
            os_mask = np.logical_and(neigh_labels != sample_label, neigh_centers == sample_center)

            so_i = float(weights[so_mask].sum())
            os_i = float(weights[os_mask].sum())
            sample_scores[i] = _ratio_or_default(so_i, os_i)

            so_total += so_i
            os_total += os_i

        return _ratio_or_default(so_total, os_total), sample_scores

    @classmethod
    def _compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        mode: str,
        k_candidates: list[int] | tuple[int, ...],
        max_pairs: int | None = None,
        random_state: int = 0,
        **kwargs: float,
    ) -> RobustnessResult:
        mode_value = str(mode).strip().lower()
        if mode_value not in cls._PAIR_MODES:
            raise ValueError(
                "mode must be one of {'paired', 'global'}"
            )

        cls._validate_inputs(features, manifest)

        df = manifest.reset_index(drop=True).copy()
        dataset_name = cls._infer_dataset_name(df)

        k = _optimal_k_by_knn_balanced_accuracy(
            features=features,
            labels=pd.factorize(df["label"])[0].astype(int),
            slide_ids=df["slide_id"].astype(str).to_numpy(),
            k_values=k_candidates,
            warn_context=f"{dataset_name} k-selection",
        )

        if mode_value == "paired":
            pairs = infer_2x2_pairs(
                df,
                dataset_name=dataset_name,
                max_pairs=max_pairs,
                random_state=random_state,
            )
            if not pairs:
                raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for RI/MaRI")
            subsets = [subset_by_pair(df, pair) for pair in pairs]
        else:
            subsets = [df]

        pair_values: list[float] = []
        sample_values: list[np.ndarray] = []
        for sub in subsets:
            if len(sub) <= 1:
                continue

            idx = sub.index.to_numpy()
            pair_features = features[idx]
            norms = np.linalg.norm(pair_features, axis=1, keepdims=True) + 1e-12
            pair_features = pair_features / norms

            labels = pd.factorize(sub["label"])[0].astype(int)
            centers = pd.factorize(sub["medical_center"])[0].astype(int)
            slide_ids = sub["slide_id"].astype(str).to_numpy()

            neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(pair_features, slide_ids, k)
            pair_value, per_sample = cls._score_from_neighbors(
                labels=labels,
                centers=centers,
                neigh_idx=neigh_idx,
                neigh_dist=neigh_dist,
                valid_counts=valid_counts,
                k=k,
                **kwargs,
            )
            pair_values.append(float(pair_value))
            sample_values.append(per_sample)

        if not pair_values:
            if mode_value == "paired":
                raise RuntimeError(f"{dataset_name}: RI/MaRI failed on all inferred 2x2 pairs")
            raise RuntimeError(f"{dataset_name}: RI/MaRI failed on full dataset")

        pair_arr = np.asarray(pair_values, dtype=float)
        if sample_values:
            sample_arr = np.concatenate(sample_values).astype(float)
        else:
            sample_arr = np.empty((0,), dtype=float)

        return RobustnessResult(
            dataset=dataset_name,
            k=int(k),
            value=float(pair_arr.mean()),
            std=float(pair_arr.std(ddof=0)),
            n_pairs=int(len(pair_arr)),
            pair_values=pair_arr,
            sample_values=sample_arr,
        )
