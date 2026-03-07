import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mari.metrics.neighbors import (
    _knn_balanced_accuracy_by_k,
    _normalize_k_values,
    _prepare_neighbors,
    _select_k_from_balanced_accuracy,
)
from mari.metrics.pairs import (
    EvaluationSubset,
    ensure_required_columns,
    normalize_center_values,
    resolve_manifest_subsets,
    validate_subset_manifest,
)
from mari.types import RobustnessResult


EVALUATION_DESIGN_PAIRED_2X2 = "paired_2x2"
EVALUATION_DESIGN_DATASET_WIDE = "dataset_wide"
VALID_EVALUATION_DESIGNS = (
    EVALUATION_DESIGN_PAIRED_2X2,
    EVALUATION_DESIGN_DATASET_WIDE,
)


def _ratio_or_default(so: float, os: float, default: float = 0.5) -> float:
    denom = float(so + os)
    if denom <= 0:
        return float(default)
    return float(float(so) / denom)


def _normalize_evaluation_design(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in VALID_EVALUATION_DESIGNS:
        raise ValueError(f"evaluation_design must be one of {list(VALID_EVALUATION_DESIGNS)}")
    return normalized


@dataclass(frozen=True)
class _PreparedSubsetInputs:
    subset_id: str
    source_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    slide_ids: np.ndarray


@dataclass(frozen=True)
class _UndefinedBreakdown:
    total_frac: float
    ss_frac: float
    oo_frac: float
    mixed_frac: float


class BaseRobustnessIndex(ABC):
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
        n = len(labels)
        target_k = int(k)
        eff_k = np.minimum(valid_counts, target_k)

        col_indices = np.arange(min(target_k, neigh_idx.shape[1]))[np.newaxis, :]
        slot_valid = (col_indices < eff_k[:, np.newaxis]) & (neigh_idx[:, :target_k] >= 0)

        idx = neigh_idx[:, :target_k]
        dist = neigh_dist[:, :target_k]
        safe_idx = np.where(slot_valid, idx, 0)

        neigh_labels = labels[safe_idx]
        neigh_centers = centers[safe_idx]

        same_label = neigh_labels == labels[:, np.newaxis]
        same_center = neigh_centers == centers[:, np.newaxis]

        so_mask = slot_valid & same_label & ~same_center
        os_mask = slot_valid & ~same_label & same_center
        ss_mask = slot_valid & same_label & same_center
        oo_mask = slot_valid & ~same_label & ~same_center

        weights = cls._weights(dist, **kwargs)
        so_per_sample = np.where(so_mask, weights, 0.0).sum(axis=1)
        os_per_sample = np.where(os_mask, weights, 0.0).sum(axis=1)

        denom = so_per_sample + os_per_sample
        informative = denom > 0
        sample_scores = np.full(n, np.nan, dtype=float)
        sample_scores[informative] = so_per_sample[informative] / denom[informative]

        undefined_type = np.zeros(n, dtype=int)
        undef = ~informative
        has_neighbors = eff_k > 0
        undef_with_neigh = undef & has_neighbors
        undef_no_neigh = undef & ~has_neighbors

        if undef_with_neigh.any():
            ss_count = ss_mask[undef_with_neigh].sum(axis=1)
            oo_count = oo_mask[undef_with_neigh].sum(axis=1)
            undefined_type[undef_with_neigh] = np.where(
                ss_count > oo_count,
                1,
                np.where(oo_count > ss_count, 2, 3),
            )
        undefined_type[undef_no_neigh] = 3

        pooled = _ratio_or_default(float(so_per_sample.sum()), float(os_per_sample.sum()))
        return pooled, sample_scores, informative, undefined_type

    @classmethod
    def _score_all_k_from_neighbors(
        cls,
        labels: np.ndarray,
        centers: np.ndarray,
        neigh_idx: np.ndarray,
        neigh_dist: np.ndarray,
        valid_counts: np.ndarray,
        k_values: list[int],
        **kwargs: float,
    ) -> dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        n = len(labels)
        kmax = int(max(k_values))
        actual_cols = min(kmax, neigh_idx.shape[1])

        col_indices = np.arange(actual_cols)[np.newaxis, :]
        eff_k_max = np.minimum(valid_counts, kmax)
        slot_valid = (col_indices < eff_k_max[:, np.newaxis]) & (neigh_idx[:, :actual_cols] >= 0)

        idx = neigh_idx[:, :actual_cols]
        dist = neigh_dist[:, :actual_cols]
        safe_idx = np.where(slot_valid, idx, 0)

        neigh_labels = labels[safe_idx]
        neigh_centers = centers[safe_idx]

        same_label = neigh_labels == labels[:, np.newaxis]
        same_center = neigh_centers == centers[:, np.newaxis]

        so_mask = slot_valid & same_label & ~same_center
        os_mask = slot_valid & ~same_label & same_center
        ss_mask = slot_valid & same_label & same_center
        oo_mask = slot_valid & ~same_label & ~same_center

        weights = cls._weights(dist, **kwargs)
        so_weighted = np.where(so_mask, weights, 0.0)
        os_weighted = np.where(os_mask, weights, 0.0)

        so_cum = np.cumsum(so_weighted, axis=1)
        os_cum = np.cumsum(os_weighted, axis=1)
        ss_cum = np.cumsum(ss_mask.astype(int), axis=1)
        oo_cum = np.cumsum(oo_mask.astype(int), axis=1)

        out: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for k in k_values:
            ki = min(int(k), actual_cols) - 1
            if ki < 0:
                out[int(k)] = (
                    0.5,
                    np.full(n, np.nan, dtype=float),
                    np.zeros(n, dtype=bool),
                    np.full(n, 3, dtype=int),
                    np.zeros(n, dtype=float),
                    np.zeros(n, dtype=float),
                )
                continue

            so_at_k = so_cum[:, ki]
            os_at_k = os_cum[:, ki]
            denom = so_at_k + os_at_k
            informative = denom > 0

            sample_scores = np.full(n, np.nan, dtype=float)
            sample_scores[informative] = so_at_k[informative] / denom[informative]

            eff_k_this = np.minimum(valid_counts, int(k))
            undefined_type = np.zeros(n, dtype=int)
            undef = ~informative
            has_neighbors = eff_k_this > 0
            undef_with_neigh = undef & has_neighbors
            undef_no_neigh = undef & ~has_neighbors

            if undef_with_neigh.any():
                ss_count = ss_cum[undef_with_neigh, ki]
                oo_count = oo_cum[undef_with_neigh, ki]
                undefined_type[undef_with_neigh] = np.where(
                    ss_count > oo_count,
                    1,
                    np.where(oo_count > ss_count, 2, 3),
                )
            undefined_type[undef_no_neigh] = 3

            pooled = _ratio_or_default(float(so_at_k.sum()), float(os_at_k.sum()))
            out[int(k)] = (pooled, sample_scores, informative, undefined_type, so_at_k, os_at_k)

        return out

    @classmethod
    def _prepare_subset_inputs(
        cls,
        *,
        features: np.ndarray,
        subset: EvaluationSubset,
    ) -> _PreparedSubsetInputs | None:
        subset_rows = subset.rows
        if len(subset_rows) <= 1:
            return None

        source_indices = subset_rows["source_sample_index"].to_numpy(dtype=int)
        subset_features = features[source_indices]
        subset_features = subset_features / (np.linalg.norm(subset_features, axis=1, keepdims=True) + 1e-12)

        return _PreparedSubsetInputs(
            subset_id=str(subset.subset_id),
            source_indices=source_indices,
            features=subset_features,
            labels=pd.factorize(subset_rows["label"])[0].astype(int),
            centers=pd.factorize(subset_rows["medical_center"])[0].astype(int),
            slide_ids=subset_rows["slide_id"].astype(str).to_numpy(),
        )

    @classmethod
    def _select_subset_k(
        cls,
        *,
        features: np.ndarray,
        subsets: list[EvaluationSubset],
        k_candidates: list[int],
        dataset_name: str,
    ) -> int:
        all_scores: list[dict[int, float]] = []
        for subset in subsets:
            prepared = cls._prepare_subset_inputs(features=features, subset=subset)
            if prepared is None:
                continue
            sub_candidates = [int(k) for k in k_candidates if int(k) < len(prepared.source_indices)]
            if not sub_candidates:
                continue
            scores = _knn_balanced_accuracy_by_k(
                features=prepared.features,
                labels=prepared.labels,
                slide_ids=prepared.slide_ids,
                k_values=sub_candidates,
                warn_context=f"{dataset_name} subset k-selection",
            )
            all_scores.append(scores)

        if not all_scores:
            raise RuntimeError(f"{dataset_name}: subset k-selection failed on all subsets")

        all_k = sorted({k for scores in all_scores for k in scores})
        averaged: dict[int, float] = {}
        for k in all_k:
            vals = [scores[k] for scores in all_scores if k in scores]
            if vals:
                averaged[k] = float(np.mean(vals))
        return _select_k_from_balanced_accuracy(k_values=all_k, scores=averaged)

    @classmethod
    def _warn_undefined_occurrences(
        cls,
        *,
        dataset_name: str,
        evaluation_unit: str,
        undefined_frac: float,
        ss_dominated_undefined_frac: float,
        oo_dominated_undefined_frac: float,
        mixed_undefined_frac: float,
    ) -> None:
        unit_label = "subset occurrences" if str(evaluation_unit) == "occurrence" else "samples"
        if undefined_frac > 0.0:
            warnings.warn(
                f"{dataset_name}: RI/MaRI undefined coverage is {undefined_frac * 100.0:.1f}% across {unit_label}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if oo_dominated_undefined_frac > max(ss_dominated_undefined_frac, mixed_undefined_frac, 0.0):
            warnings.warn(
                f"{dataset_name}: undefined RI/MaRI {unit_label} are predominantly OO-dominated ({oo_dominated_undefined_frac * 100.0:.1f}%).",
                RuntimeWarning,
                stacklevel=2,
            )

    @classmethod
    def _compute_undefined_breakdown(
        cls,
        *,
        occurrence_total: int,
        ss_undefined: int,
        oo_undefined: int,
        mixed_undefined: int,
    ) -> _UndefinedBreakdown:
        if occurrence_total <= 0:
            return _UndefinedBreakdown(total_frac=0.0, ss_frac=0.0, oo_frac=0.0, mixed_frac=0.0)

        total_undefined = ss_undefined + oo_undefined + mixed_undefined
        denominator = float(occurrence_total)
        return _UndefinedBreakdown(
            total_frac=float(total_undefined / denominator),
            ss_frac=float(ss_undefined / denominator),
            oo_frac=float(oo_undefined / denominator),
            mixed_frac=float(mixed_undefined / denominator),
        )

    @classmethod
    def _compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        k_candidates: list[int] | tuple[int, ...],
        evaluation_design: str = EVALUATION_DESIGN_PAIRED_2X2,
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
        **kwargs: float,
    ) -> RobustnessResult:
        cls._validate_inputs(features, manifest)

        df = manifest.reset_index(drop=True).copy()
        dataset_name = cls._infer_dataset_name(df)
        evaluation_design = _normalize_evaluation_design(evaluation_design)
        if evaluation_design == EVALUATION_DESIGN_PAIRED_2X2:
            validate_subset_manifest(df, f"manifest for dataset '{dataset_name}'")
        features, df = cls._apply_center_exclusion(
            features=features,
            manifest=df,
            dataset_name=dataset_name,
            exclude_centers=exclude_centers,
        )
        candidates = _normalize_k_values(k_candidates)

        if evaluation_design == EVALUATION_DESIGN_DATASET_WIDE:
            return cls._compute_dataset_wide(
                features=features,
                df=df,
                dataset_name=dataset_name,
                k_candidates=candidates,
                **kwargs,
            )

        subsets = cls._build_subsets(
            df=df,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            random_state=random_state,
        )
        k = cls._select_subset_k(
            features=features,
            subsets=subsets,
            k_candidates=candidates,
            dataset_name=dataset_name,
        )
        by_k = cls._score_subsets_by_k(
            features=features,
            subsets=subsets,
            k_values=[int(k)],
            dataset_name=dataset_name,
            **kwargs,
        )
        (
            pooled,
            pair_arr,
            sample_arr,
            occurrence_values,
            occurrence_defined_mask,
            undef_type_arr,
            occurrence_subsets,
            occurrence_source_indices,
            undefined_frac,
            ss_dominated_undefined_frac,
            oo_dominated_undefined_frac,
            mixed_undefined_frac,
        ) = by_k[int(k)]

        cls._warn_undefined_occurrences(
            dataset_name=dataset_name,
            evaluation_unit="occurrence",
            undefined_frac=undefined_frac,
            ss_dominated_undefined_frac=ss_dominated_undefined_frac,
            oo_dominated_undefined_frac=oo_dominated_undefined_frac,
            mixed_undefined_frac=mixed_undefined_frac,
        )

        return RobustnessResult(
            dataset=dataset_name,
            k=int(k),
            value=float(pooled),
            std=float(pair_arr.std(ddof=0)),
            n_pairs=int(len(pair_arr)),
            pair_values=pair_arr,
            sample_values=sample_arr,
            sample_values_aligned=occurrence_values,
            occurrence_defined_mask=occurrence_defined_mask,
            sample_undefined_types=undef_type_arr,
            occurrence_subsets=occurrence_subsets,
            occurrence_source_indices=occurrence_source_indices,
            undefined_frac=undefined_frac,
            ss_dominated_undefined_frac=ss_dominated_undefined_frac,
            oo_dominated_undefined_frac=oo_dominated_undefined_frac,
            mixed_undefined_frac=mixed_undefined_frac,
            evaluation_design=evaluation_design,
            evaluation_unit="occurrence",
        )

    @classmethod
    def _compute_curve(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        k_values: list[int] | tuple[int, ...],
        evaluation_design: str = EVALUATION_DESIGN_PAIRED_2X2,
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
        **kwargs: float,
    ) -> dict[int, float]:
        cls._validate_inputs(features, manifest)

        df = manifest.reset_index(drop=True).copy()
        dataset_name = cls._infer_dataset_name(df)
        evaluation_design = _normalize_evaluation_design(evaluation_design)
        if evaluation_design == EVALUATION_DESIGN_PAIRED_2X2:
            validate_subset_manifest(df, f"manifest for dataset '{dataset_name}'")
        features, df = cls._apply_center_exclusion(
            features=features,
            manifest=df,
            dataset_name=dataset_name,
            exclude_centers=exclude_centers,
        )
        candidates = _normalize_k_values(k_values)
        if evaluation_design == EVALUATION_DESIGN_DATASET_WIDE:
            return cls._compute_curve_dataset_wide(
                features=features,
                df=df,
                dataset_name=dataset_name,
                k_values=candidates,
                **kwargs,
            )
        subsets = cls._build_subsets(
            df=df,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            random_state=random_state,
        )
        by_k = cls._score_subsets_by_k(
            features=features,
            subsets=subsets,
            k_values=candidates,
            dataset_name=dataset_name,
            **kwargs,
        )
        return {int(k): float(by_k[int(k)][0]) for k in candidates if int(k) in by_k}

    @classmethod
    def _prepare_dataset_wide_inputs(
        cls,
        *,
        features: np.ndarray,
        df: pd.DataFrame,
    ) -> _PreparedSubsetInputs:
        source_indices = np.arange(len(df), dtype=int)
        normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        return _PreparedSubsetInputs(
            subset_id="dataset",
            source_indices=source_indices,
            features=normalized_features,
            labels=pd.factorize(df["label"])[0].astype(int),
            centers=pd.factorize(df["medical_center"])[0].astype(int),
            slide_ids=df["slide_id"].astype(str).to_numpy(),
        )

    @classmethod
    def _select_dataset_wide_k(
        cls,
        *,
        prepared: _PreparedSubsetInputs,
        k_candidates: list[int],
        dataset_name: str,
    ) -> int:
        valid_candidates = [int(k) for k in k_candidates if int(k) < len(prepared.source_indices)]
        if not valid_candidates:
            raise RuntimeError(f"{dataset_name}: dataset-wide k-selection failed because no valid k candidates remain")
        scores = _knn_balanced_accuracy_by_k(
            features=prepared.features,
            labels=prepared.labels,
            slide_ids=prepared.slide_ids,
            k_values=valid_candidates,
            warn_context=f"{dataset_name} dataset-wide k-selection",
        )
        return _select_k_from_balanced_accuracy(k_values=valid_candidates, scores=scores)

    @classmethod
    def _score_dataset_wide_by_k(
        cls,
        *,
        prepared: _PreparedSubsetInputs,
        k_values: list[int] | tuple[int, ...],
        dataset_name: str,
        **kwargs: float,
    ) -> dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]]:
        candidates = _normalize_k_values(k_values)
        kmax = int(max(candidates))
        neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(prepared.features, prepared.slide_ids, kmax)
        all_k_results = cls._score_all_k_from_neighbors(
            labels=prepared.labels,
            centers=prepared.centers,
            neigh_idx=neigh_idx,
            neigh_dist=neigh_dist,
            valid_counts=valid_counts,
            k_values=candidates,
            **kwargs,
        )
        out: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]] = {}
        occurrence_subsets = np.full(len(prepared.source_indices), prepared.subset_id, dtype=object).astype(str)
        occurrence_sources = prepared.source_indices.astype(int)
        for k in candidates:
            pooled, per_sample, informative_mask, undefined_type, _so_arr, _os_arr = all_k_results[int(k)]
            sample_arr = np.asarray(per_sample[informative_mask], dtype=float)
            undefined_breakdown = cls._compute_undefined_breakdown(
                occurrence_total=len(per_sample),
                ss_undefined=int(np.count_nonzero(undefined_type == 1)),
                oo_undefined=int(np.count_nonzero(undefined_type == 2)),
                mixed_undefined=int(np.count_nonzero(undefined_type == 3)),
            )
            pair_arr = np.asarray([pooled], dtype=float)
            out[int(k)] = (
                float(pooled),
                pair_arr,
                sample_arr,
                np.asarray(per_sample, dtype=float),
                np.asarray(informative_mask, dtype=bool),
                np.asarray(undefined_type, dtype=int),
                occurrence_subsets,
                occurrence_sources,
                undefined_breakdown.total_frac,
                undefined_breakdown.ss_frac,
                undefined_breakdown.oo_frac,
                undefined_breakdown.mixed_frac,
            )
        if not out:
            raise RuntimeError(f"{dataset_name}: RI/MaRI failed on the dataset-wide evaluation design")
        return out

    @classmethod
    def _compute_dataset_wide(
        cls,
        *,
        features: np.ndarray,
        df: pd.DataFrame,
        dataset_name: str,
        k_candidates: list[int],
        **kwargs: float,
    ) -> RobustnessResult:
        prepared = cls._prepare_dataset_wide_inputs(features=features, df=df)
        k = cls._select_dataset_wide_k(
            prepared=prepared,
            k_candidates=k_candidates,
            dataset_name=dataset_name,
        )
        by_k = cls._score_dataset_wide_by_k(
            prepared=prepared,
            k_values=[int(k)],
            dataset_name=dataset_name,
            **kwargs,
        )
        (
            pooled,
            pair_arr,
            sample_arr,
            sample_values_aligned,
            occurrence_defined_mask,
            undefined_types,
            occurrence_subsets,
            occurrence_source_indices,
            undefined_frac,
            ss_dominated_undefined_frac,
            oo_dominated_undefined_frac,
            mixed_undefined_frac,
        ) = by_k[int(k)]

        cls._warn_undefined_occurrences(
            dataset_name=dataset_name,
            evaluation_unit="sample",
            undefined_frac=undefined_frac,
            ss_dominated_undefined_frac=ss_dominated_undefined_frac,
            oo_dominated_undefined_frac=oo_dominated_undefined_frac,
            mixed_undefined_frac=mixed_undefined_frac,
        )

        return RobustnessResult(
            dataset=dataset_name,
            k=int(k),
            value=float(pooled),
            std=float(pair_arr.std(ddof=0)),
            n_pairs=1,
            pair_values=pair_arr,
            sample_values=sample_arr,
            sample_values_aligned=sample_values_aligned,
            occurrence_defined_mask=occurrence_defined_mask,
            sample_undefined_types=undefined_types,
            occurrence_subsets=occurrence_subsets,
            occurrence_source_indices=occurrence_source_indices,
            undefined_frac=undefined_frac,
            ss_dominated_undefined_frac=ss_dominated_undefined_frac,
            oo_dominated_undefined_frac=oo_dominated_undefined_frac,
            mixed_undefined_frac=mixed_undefined_frac,
            evaluation_design=EVALUATION_DESIGN_DATASET_WIDE,
            evaluation_unit="sample",
        )

    @classmethod
    def _compute_curve_dataset_wide(
        cls,
        *,
        features: np.ndarray,
        df: pd.DataFrame,
        dataset_name: str,
        k_values: list[int],
        **kwargs: float,
    ) -> dict[int, float]:
        prepared = cls._prepare_dataset_wide_inputs(features=features, df=df)
        by_k = cls._score_dataset_wide_by_k(
            prepared=prepared,
            k_values=k_values,
            dataset_name=dataset_name,
            **kwargs,
        )
        return {int(k): float(by_k[int(k)][0]) for k in k_values if int(k) in by_k}

    @classmethod
    def _build_subsets(
        cls,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        max_pairs: int | None,
        random_state: int,
    ) -> list[EvaluationSubset]:
        del max_pairs, random_state
        subsets = resolve_manifest_subsets(df)
        if not subsets:
            raise RuntimeError(f"{dataset_name}: no valid manifest-defined 2x2 subsets remain for RI/MaRI")
        return subsets

    @classmethod
    def _score_subsets_by_k(
        cls,
        *,
        features: np.ndarray,
        subsets: list[EvaluationSubset],
        k_values: list[int] | tuple[int, ...],
        dataset_name: str,
        **kwargs: float,
    ) -> dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]]:
        candidates = _normalize_k_values(k_values)
        kmax = int(max(candidates))

        per_k_pair_values: dict[int, list[float]] = {int(k): [] for k in candidates}
        per_k_so_total: dict[int, float] = {int(k): 0.0 for k in candidates}
        per_k_os_total: dict[int, float] = {int(k): 0.0 for k in candidates}
        per_k_occurrence_values: dict[int, list[np.ndarray]] = {int(k): [] for k in candidates}
        per_k_occurrence_defined: dict[int, list[np.ndarray]] = {int(k): [] for k in candidates}
        per_k_occurrence_types: dict[int, list[np.ndarray]] = {int(k): [] for k in candidates}
        per_k_occurrence_subsets: dict[int, list[np.ndarray]] = {int(k): [] for k in candidates}
        per_k_occurrence_sources: dict[int, list[np.ndarray]] = {int(k): [] for k in candidates}
        per_k_undefined_counts: dict[int, dict[int, int]] = {int(k): {1: 0, 2: 0, 3: 0} for k in candidates}
        per_k_occurrence_total: dict[int, int] = {int(k): 0 for k in candidates}

        for subset in subsets:
            prepared = cls._prepare_subset_inputs(features=features, subset=subset)
            if prepared is None:
                continue

            neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(
                prepared.features,
                prepared.slide_ids,
                kmax,
            )
            all_k_results = cls._score_all_k_from_neighbors(
                labels=prepared.labels,
                centers=prepared.centers,
                neigh_idx=neigh_idx,
                neigh_dist=neigh_dist,
                valid_counts=valid_counts,
                k_values=candidates,
                **kwargs,
            )

            for k in candidates:
                pair_value, per_sample, informative_mask, undefined_type, so_arr, os_arr = all_k_results[int(k)]
                per_k_pair_values[int(k)].append(float(pair_value))
                per_k_occurrence_total[int(k)] += int(len(per_sample))
                per_k_so_total[int(k)] += float(so_arr.sum())
                per_k_os_total[int(k)] += float(os_arr.sum())
                per_k_occurrence_values[int(k)].append(np.asarray(per_sample, dtype=float))
                per_k_occurrence_defined[int(k)].append(np.asarray(informative_mask, dtype=bool))
                per_k_occurrence_types[int(k)].append(np.asarray(undefined_type, dtype=int))
                per_k_occurrence_subsets[int(k)].append(np.full(len(per_sample), prepared.subset_id, dtype=object))
                per_k_occurrence_sources[int(k)].append(prepared.source_indices)
                per_k_undefined_counts[int(k)][1] += int(np.count_nonzero(undefined_type == 1))
                per_k_undefined_counts[int(k)][2] += int(np.count_nonzero(undefined_type == 2))
                per_k_undefined_counts[int(k)][3] += int(np.count_nonzero(undefined_type == 3))

        if not any(per_k_pair_values[int(k)] for k in candidates):
            raise RuntimeError(f"{dataset_name}: RI/MaRI failed on all manifest-defined subsets")

        out: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]] = {}
        for k in candidates:
            pair_values = per_k_pair_values[int(k)]
            if not pair_values:
                continue

            pooled = _ratio_or_default(per_k_so_total[int(k)], per_k_os_total[int(k)])
            pair_arr = np.asarray(pair_values, dtype=float)
            occurrence_values = np.concatenate(per_k_occurrence_values[int(k)]).astype(float)
            occurrence_defined_mask = np.concatenate(per_k_occurrence_defined[int(k)]).astype(bool)
            undef_type_arr = np.concatenate(per_k_occurrence_types[int(k)]).astype(int)
            occurrence_subsets = np.concatenate(per_k_occurrence_subsets[int(k)]).astype(str)
            occurrence_source_indices = np.concatenate(per_k_occurrence_sources[int(k)]).astype(int)
            sample_arr = occurrence_values[occurrence_defined_mask].astype(float)

            undefined_breakdown = cls._compute_undefined_breakdown(
                occurrence_total=int(per_k_occurrence_total[int(k)]),
                ss_undefined=int(per_k_undefined_counts[int(k)][1]),
                oo_undefined=int(per_k_undefined_counts[int(k)][2]),
                mixed_undefined=int(per_k_undefined_counts[int(k)][3]),
            )
            out[int(k)] = (
                pooled,
                pair_arr,
                sample_arr,
                occurrence_values,
                occurrence_defined_mask,
                undef_type_arr,
                occurrence_subsets,
                occurrence_source_indices,
                undefined_breakdown.total_frac,
                undefined_breakdown.ss_frac,
                undefined_breakdown.oo_frac,
                undefined_breakdown.mixed_frac,
            )
        return out
