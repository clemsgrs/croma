
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from mari.metrics.neighbors import _normalize_k_values, _optimal_k_by_knn_balanced_accuracy, _prepare_neighbors
from mari.metrics.pairs import ensure_required_columns, infer_2x2_pairs, normalize_center_values, subset_by_pair
from mari.types import RobustnessResult


def _ratio_or_default(so: float, os: float, default: float = 0.5) -> float:
    denom = float(so + os)
    if denom <= 0:
        return float(default)
    return float(float(so) / denom)


def _ratio_or_nan(so: float, os: float) -> float:
    denom = float(so + os)
    if denom <= 0:
        return float("nan")
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
    def _apply_center_exclusion(
        cls,
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        dataset_name: str,
        exclude_centers: object | None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        excluded = normalize_center_values(exclude_centers)
        if not excluded:
            return features, manifest

        center_series = manifest["medical_center"].map(lambda v: str(v).strip())
        keep_mask = ~center_series.isin(excluded)
        if not bool(keep_mask.any()):
            excluded_txt = ", ".join(excluded)
            raise ValueError(
                f"No samples remain after excluding centers [{excluded_txt}] from dataset '{dataset_name}'"
            )

        kept_features = features[keep_mask.to_numpy()]
        kept_manifest = manifest.loc[keep_mask].reset_index(drop=True)
        return kept_features, kept_manifest

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
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        target_k = int(k)
        so_total = 0.0
        os_total = 0.0
        sample_scores = np.full((len(labels),), np.nan, dtype=float)
        informative_mask = np.zeros((len(labels),), dtype=bool)
        # 0=defined, 1=SS-dominated, 2=OO-dominated, 3=mixed / unclassified
        undefined_type = np.zeros((len(labels),), dtype=int)

        for i in range(len(labels)):
            eff_k = min(target_k, int(valid_counts[i]))
            if eff_k <= 0:
                undefined_type[i] = 3
                continue

            row_idx = neigh_idx[i, :eff_k]
            row_dist = neigh_dist[i, :eff_k]
            keep = row_idx >= 0
            if not bool(np.any(keep)):
                undefined_type[i] = 3
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
            sample_score = _ratio_or_nan(so_i, os_i)
            if np.isfinite(sample_score):
                sample_scores[i] = float(sample_score)
                informative_mask[i] = True
            else:
                ss_count = int(np.logical_and(neigh_labels == sample_label, neigh_centers == sample_center).sum())
                oo_count = int(np.logical_and(neigh_labels != sample_label, neigh_centers != sample_center).sum())
                if ss_count > oo_count:
                    undefined_type[i] = 1  # SS-dominated
                elif oo_count > ss_count:
                    undefined_type[i] = 2  # OO-dominated
                else:
                    undefined_type[i] = 3  # mixed

            so_total += so_i
            os_total += os_i

        return _ratio_or_default(so_total, os_total), sample_scores, informative_mask, undefined_type

    @classmethod
    def _compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        mode: str,
        k_candidates: list[int] | tuple[int, ...],
        exclude_centers: object | None = None,
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
        features, df = cls._apply_center_exclusion(
            features=features,
            manifest=df,
            dataset_name=dataset_name,
            exclude_centers=exclude_centers,
        )
        candidates = _normalize_k_values(k_candidates)

        k = _optimal_k_by_knn_balanced_accuracy(
            features=features,
            labels=pd.factorize(df["label"])[0].astype(int),
            slide_ids=df["slide_id"].astype(str).to_numpy(),
            k_values=candidates,
            warn_context=f"{dataset_name} k-selection",
        )

        subsets = cls._build_subsets(
            df=df,
            mode_value=mode_value,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            random_state=random_state,
        )
        by_k = cls._score_subsets_by_k(
            features=features,
            subsets=subsets,
            k_values=[int(k)],
            mode_value=mode_value,
            dataset_name=dataset_name,
            **kwargs,
        )
        pair_arr, sample_arr, sample_aligned, undef_type_arr = by_k[int(k)]

        total_n = len(features)
        informative_n = len(sample_arr)
        undefined_n = max(0, total_n - informative_n)
        undefined_frac = float(undefined_n / total_n) if total_n > 0 else 0.0
        if mode_value == "paired":
            warnings.warn(
                "RI/MaRI undefined subtype breakdown is only well-defined in global mode; "
                "paired-mode subtype fractions are reported as NaN until paired aggregation semantics are specified.",
                RuntimeWarning,
                stacklevel=2,
            )
            ss_dominated_undefined_frac = float("nan")
            oo_dominated_undefined_frac = float("nan")
            mixed_undefined_frac = float("nan")
        else:
            ss_count = int((undef_type_arr == 1).sum())
            oo_count = int((undef_type_arr == 2).sum())
            mixed_count = int((undef_type_arr == 3).sum())
            ss_dominated_undefined_frac = float(ss_count / total_n) if total_n > 0 else 0.0
            oo_dominated_undefined_frac = float(oo_count / total_n) if total_n > 0 else 0.0
            mixed_undefined_frac = float(mixed_count / total_n) if total_n > 0 else 0.0

        return RobustnessResult(
            dataset=dataset_name,
            k=int(k),
            value=float(pair_arr.mean()),
            std=float(pair_arr.std(ddof=0)),
            n_pairs=int(len(pair_arr)),
            pair_values=pair_arr,
            sample_values=sample_arr,
            sample_values_aligned=sample_aligned,
            sample_undefined_types=undef_type_arr,
            undefined_frac=undefined_frac,
            ss_dominated_undefined_frac=ss_dominated_undefined_frac,
            oo_dominated_undefined_frac=oo_dominated_undefined_frac,
            mixed_undefined_frac=mixed_undefined_frac,
        )

    @classmethod
    def _compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        mode: str,
        k_values: list[int] | tuple[int, ...],
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
        **kwargs: float,
    ) -> dict[int, float]:
        mode_value = str(mode).strip().lower()
        if mode_value not in cls._PAIR_MODES:
            raise ValueError(
                "mode must be one of {'paired', 'global'}"
            )
        cls._validate_inputs(features, manifest)

        df = manifest.reset_index(drop=True).copy()
        dataset_name = cls._infer_dataset_name(df)
        features, df = cls._apply_center_exclusion(
            features=features,
            manifest=df,
            dataset_name=dataset_name,
            exclude_centers=exclude_centers,
        )
        candidates = _normalize_k_values(k_values)
        subsets = cls._build_subsets(
            df=df,
            mode_value=mode_value,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            random_state=random_state,
        )
        by_k = cls._score_subsets_by_k(
            features=features,
            subsets=subsets,
            k_values=candidates,
            mode_value=mode_value,
            dataset_name=dataset_name,
            **kwargs,
        )
        return {int(k): float(by_k[int(k)][0].mean()) for k in candidates if int(k) in by_k}

    @classmethod
    def _build_subsets(
        cls,
        *,
        df: pd.DataFrame,
        mode_value: str,
        dataset_name: str,
        max_pairs: int | None,
        random_state: int,
    ) -> list[pd.DataFrame]:
        if mode_value == "paired":
            pairs = infer_2x2_pairs(
                df,
                dataset_name=dataset_name,
                max_pairs=max_pairs,
                random_state=random_state,
            )
            if not pairs:
                raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for RI/MaRI")
            return [subset_by_pair(df, pair) for pair in pairs]
        return [df]

    @classmethod
    def _score_subsets_by_k(
        cls,
        *,
        features: np.ndarray,
        subsets: list[pd.DataFrame],
        k_values: list[int] | tuple[int, ...],
        mode_value: str,
        dataset_name: str,
        **kwargs: float,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        candidates = _normalize_k_values(k_values)
        kmax = int(max(candidates))
        per_k_pair_values: dict[int, list[float]] = {int(k): [] for k in candidates}
        per_k_sample_sum: dict[int, np.ndarray] = {
            int(k): np.zeros((len(features),), dtype=float) for k in candidates
        }
        per_k_sample_count: dict[int, np.ndarray] = {
            int(k): np.zeros((len(features),), dtype=int) for k in candidates
        }
        # Track undefined type per sample (last-seen wins for multi-subset)
        per_k_undefined_type: dict[int, np.ndarray] = {
            int(k): np.zeros((len(features),), dtype=int) for k in candidates
        }

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

            neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(pair_features, slide_ids, kmax)
            for k in candidates:
                pair_value, per_sample, informative_mask, undefined_type = cls._score_from_neighbors(
                    labels=labels,
                    centers=centers,
                    neigh_idx=neigh_idx,
                    neigh_dist=neigh_dist,
                    valid_counts=valid_counts,
                    k=int(k),
                    **kwargs,
                )
                per_k_pair_values[int(k)].append(float(pair_value))
                if bool(np.any(informative_mask)):
                    global_idx = idx[informative_mask]
                    per_k_sample_sum[int(k)][global_idx] += per_sample[informative_mask]
                    per_k_sample_count[int(k)][global_idx] += 1
                # For undefined samples, store their type (using global indices)
                undefined_mask = ~informative_mask & (undefined_type > 0)
                if bool(np.any(undefined_mask)):
                    global_undef_idx = idx[undefined_mask]
                    per_k_undefined_type[int(k)][global_undef_idx] = undefined_type[undefined_mask]

        if not any(per_k_pair_values[int(k)] for k in candidates):
            if mode_value == "paired":
                raise RuntimeError(f"{dataset_name}: RI/MaRI failed on all inferred 2x2 pairs")
            raise RuntimeError(f"{dataset_name}: RI/MaRI failed on full dataset")

        out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for k in candidates:
            pair_values = per_k_pair_values[int(k)]
            if not pair_values:
                continue
            pair_arr = np.asarray(pair_values, dtype=float)
            counts = per_k_sample_count[int(k)]
            informative = counts > 0
            sample_aligned = np.full((len(features),), np.nan, dtype=float)
            if bool(np.any(informative)):
                sample_aligned[informative] = (
                    per_k_sample_sum[int(k)][informative] / counts[informative]
                ).astype(float)
                sample_arr = sample_aligned[informative].astype(float)
            else:
                sample_arr = np.empty((0,), dtype=float)
            undef_type_arr = per_k_undefined_type[int(k)]
            out[int(k)] = (pair_arr, sample_arr, sample_aligned, undef_type_arr)
        return out
