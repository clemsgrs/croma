import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from croma.metrics.base import (
    EVALUATION_DESIGN_DATASET_WIDE,
    EVALUATION_DESIGN_PAIRED_2X2,
    _normalize_evaluation_design,
)
from croma.metrics.neighbors import _filter_query_neighbors_excluding_same_slide, _initial_n_neighbors
from croma.metrics.pairs import (
    EvaluationSubset,
    ensure_required_columns,
    resolve_manifest_subsets,
    validate_subset_manifest,
)
from croma.metrics.tail import compute_tail_metrics
from croma.types import CCMRResult

logger = logging.getLogger("croma")


@dataclass(frozen=True)
class _CCMRSearchMeta:
    k_start: int
    k_final: int
    retries: int


def _compute_sample_ccmr(
    so_dists: np.ndarray,
    os_dists: np.ndarray,
) -> np.ndarray:
    has_inf_so = np.any(np.isinf(so_dists), axis=1)
    has_inf_os = np.any(np.isinf(os_dists), axis=1)
    undefined = has_inf_so | has_inf_os

    mean_so = np.mean(so_dists, axis=1)
    mean_os = np.mean(os_dists, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ccmr = mean_os / mean_so

    ccmr[undefined] = np.nan
    ccmr[mean_so == 0.0] = np.nan
    return ccmr


def _scan_typed_neighbors_for_query_rows(
    *,
    labels: np.ndarray,
    centers: np.ndarray,
    query_indices: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_dist: np.ndarray,
    valid_counts: np.ndarray,
    m: int,
    so_dists: np.ndarray,
    os_dists: np.ndarray,
) -> np.ndarray:
    defined = np.zeros((len(query_indices),), dtype=bool)
    for row, sample_idx in enumerate(query_indices.tolist()):
        so_count = 0
        os_count = 0
        so_dists[sample_idx, :] = np.inf
        os_dists[sample_idx, :] = np.inf

        eff_k = int(valid_counts[row])
        for pos in range(eff_k):
            j = int(neigh_idx[row, pos])
            if j < 0:
                continue
            d = float(neigh_dist[row, pos])
            same_label = labels[j] == labels[sample_idx]
            same_center = centers[j] == centers[sample_idx]
            if same_label and not same_center and so_count < m:
                so_dists[sample_idx, so_count] = d
                so_count += 1
            elif not same_label and same_center and os_count < m:
                os_dists[sample_idx, os_count] = d
                os_count += 1
            if so_count >= m and os_count >= m:
                defined[row] = True
                break

    return defined


def _iterative_typed_neighbor_search(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    slide_ids: np.ndarray,
    m: int,
    start_k: int,
    k_growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, _CCMRSearchMeta]:
    n_samples = int(len(labels))
    so_dists = np.full((n_samples, int(m)), np.inf, dtype=float)
    os_dists = np.full((n_samples, int(m)), np.inf, dtype=float)

    if n_samples <= 1:
        return (
            so_dists,
            os_dists,
            _CCMRSearchMeta(
                k_start=0,
                k_final=0,
                retries=0,
            ),
        )

    model = NearestNeighbors(metric="cosine")
    model.fit(features)

    k_current = int(max(1, min(int(start_k), n_samples - 1)))
    k_start_used = int(k_current)
    retries = 0

    defined_mask = np.zeros((n_samples,), dtype=bool)
    unresolved_mask = ~defined_mask

    while True:
        query_indices = np.flatnonzero(unresolved_mask)
        if int(query_indices.size) <= 0:
            return (
                so_dists,
                os_dists,
                _CCMRSearchMeta(
                    k_start=k_start_used,
                    k_final=int(k_current),
                    retries=int(retries),
                ),
            )

        fetch_neighbors = _initial_n_neighbors(kmax=int(k_current), slide_ids=slide_ids, n_samples=n_samples)
        distances, raw_neighbors = model.kneighbors(features[query_indices], n_neighbors=fetch_neighbors)

        neigh_idx, neigh_dist, valid_counts = _filter_query_neighbors_excluding_same_slide(
            raw_neighbors=raw_neighbors,
            raw_distances=distances,
            query_indices=query_indices,
            slide_ids=slide_ids,
            kmax=int(k_current),
        )

        newly_defined = _scan_typed_neighbors_for_query_rows(
            labels=labels,
            centers=centers,
            query_indices=query_indices,
            neigh_idx=neigh_idx,
            neigh_dist=neigh_dist,
            valid_counts=valid_counts,
            m=int(m),
            so_dists=so_dists,
            os_dists=os_dists,
        )
        if bool(np.any(newly_defined)):
            defined_mask[query_indices[newly_defined]] = True
        unresolved_mask = ~defined_mask

        if int(k_current) >= n_samples - 1:
            return (
                so_dists,
                os_dists,
                _CCMRSearchMeta(
                    k_start=k_start_used,
                    k_final=int(k_current),
                    retries=int(retries),
                ),
            )

        retries += 1
        grown_k = int(math.ceil(float(k_current) * float(k_growth_factor)))
        k_current = int(min(n_samples - 1, max(int(k_current) + 1, grown_k)))


def _dataset_subset(df: pd.DataFrame) -> EvaluationSubset:
    subset_df = df.copy()
    subset_df["source_sample_index"] = subset_df.index.astype(int)
    subset_df["subset"] = "dataset"
    subset_df = subset_df.reset_index(drop=True)
    return EvaluationSubset(subset_id="dataset", rows=subset_df)


class CrossConfounderMarginRatio:
    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        evaluation_design: str = EVALUATION_DESIGN_PAIRED_2X2,
        m: int | list[int] | tuple[int, ...] = 1,
        alpha: float = 0.10,
        start_k: int = 200,
        k_growth_factor: float = 2.0,
    ) -> CCMRResult | dict[int, CCMRResult]:
        evaluation_design = _normalize_evaluation_design(evaluation_design)

        if isinstance(m, (list, tuple)):
            if len(m) <= 0:
                raise ValueError("m must include at least one integer")
            ordered_m_values = [int(v) for v in m]
            return_single = False
        else:
            ordered_m_values = [int(m)]
            return_single = True
        unique_m_values = sorted(set(ordered_m_values))
        if min(unique_m_values) < 1:
            raise ValueError("m must be >= 1")
        if int(start_k) < 1:
            raise ValueError("start_k must be >= 1")
        if float(k_growth_factor) <= 1.0:
            raise ValueError("k_growth_factor must be > 1")

        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas.DataFrame")
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a numpy.ndarray")
        if features.ndim != 2:
            raise ValueError("features must be a 2-D array of shape (N, D)")
        if len(features) != len(manifest):
            raise ValueError("features row count must match manifest row count")
        ensure_required_columns(manifest, "manifest")

        df = manifest.reset_index(drop=True).copy()
        dataset_name = cls._infer_dataset_name(df)
        if evaluation_design == EVALUATION_DESIGN_PAIRED_2X2:
            validate_subset_manifest(df, f"manifest for dataset '{dataset_name}'")

        if evaluation_design == EVALUATION_DESIGN_PAIRED_2X2:
            subsets = resolve_manifest_subsets(df)
            if not subsets:
                raise RuntimeError(f"{dataset_name}: no valid manifest-defined 2x2 subsets remain for CCMR")
            evaluation_unit = "occurrence"
        else:
            subsets = [_dataset_subset(df)]
            evaluation_unit = "sample"

        pair_medians: dict[int, list[float]] = {int(mm): [] for mm in unique_m_values}
        occurrence_values_by_m: dict[int, list[np.ndarray]] = {int(mm): [] for mm in unique_m_values}
        occurrence_subsets_by_m: dict[int, list[np.ndarray]] = {int(mm): [] for mm in unique_m_values}
        occurrence_sources_by_m: dict[int, list[np.ndarray]] = {int(mm): [] for mm in unique_m_values}
        occurrence_total = 0
        total_undefined: dict[int, int] = {int(mm): 0 for mm in unique_m_values}

        k_start_values: list[int] = []
        k_final_values: list[int] = []
        retries_values: list[int] = []
        m_max = int(max(unique_m_values))

        for subset in subsets:
            sub = subset.rows
            if len(sub) <= 1:
                continue

            idx = sub["source_sample_index"].to_numpy(dtype=int)
            subset_features = features[idx]
            subset_features = subset_features / (np.linalg.norm(subset_features, axis=1, keepdims=True) + 1e-12)

            labels = pd.factorize(sub["label"])[0].astype(int)
            centers = pd.factorize(sub["medical_center"])[0].astype(int)
            slide_ids = sub["slide_id"].astype(str).to_numpy()

            so_dists, os_dists, search_meta = _iterative_typed_neighbor_search(
                features=subset_features,
                labels=labels,
                centers=centers,
                slide_ids=slide_ids,
                m=int(m_max),
                start_k=int(start_k),
                k_growth_factor=float(k_growth_factor),
            )

            n_sub = len(sub)
            occurrence_total += n_sub
            k_start_values.append(int(search_meta.k_start))
            k_final_values.append(int(search_meta.k_final))
            retries_values.append(int(search_meta.retries))

            for mm in unique_m_values:
                sample_ccmr = _compute_sample_ccmr(so_dists[:, : int(mm)], os_dists[:, : int(mm)])
                informative = np.isfinite(sample_ccmr)
                n_informative = int(informative.sum())
                n_undefined = int(n_sub - n_informative)
                total_undefined[int(mm)] += n_undefined

                occurrence_values_by_m[int(mm)].append(np.asarray(sample_ccmr, dtype=float))
                occurrence_subsets_by_m[int(mm)].append(np.full(n_sub, str(subset.subset_id), dtype=object))
                occurrence_sources_by_m[int(mm)].append(idx.astype(int))

                if n_informative > 0:
                    pair_medians[int(mm)].append(float(np.median(sample_ccmr[informative])))
                else:
                    pair_medians[int(mm)].append(float("nan"))

        k_start_value = int(min(k_start_values)) if k_start_values else 0
        k_final_value = int(max(k_final_values)) if k_final_values else 0
        retries_value = int(max(retries_values)) if retries_values else 0

        by_m: dict[int, CCMRResult] = {}
        for mm in unique_m_values:
            finite_pair = np.asarray(pair_medians[int(mm)], dtype=float)
            finite_mask = np.isfinite(finite_pair)
            if finite_mask.any():
                value = float(np.median(finite_pair[finite_mask]))
                std = float(finite_pair[finite_mask].std(ddof=0)) if finite_mask.sum() > 1 else 0.0
            else:
                value = float("nan")
                std = 0.0

            occurrence_values = (
                np.concatenate(occurrence_values_by_m[int(mm)]).astype(float)
                if occurrence_values_by_m[int(mm)]
                else np.empty((0,), dtype=float)
            )
            occurrence_subsets = (
                np.concatenate(occurrence_subsets_by_m[int(mm)]).astype(str)
                if occurrence_subsets_by_m[int(mm)]
                else np.empty((0,), dtype=str)
            )
            occurrence_sources = (
                np.concatenate(occurrence_sources_by_m[int(mm)]).astype(int)
                if occurrence_sources_by_m[int(mm)]
                else np.empty((0,), dtype=int)
            )
            occurrence_defined_mask = np.isfinite(occurrence_values)
            sample_values = occurrence_values[np.isfinite(occurrence_values)].astype(float)
            undefined_frac = float(total_undefined[int(mm)] / occurrence_total) if occurrence_total > 0 else 0.0

            if total_undefined[int(mm)] > 0:
                logger.warning(
                    f"[CCMR] dataset '{dataset_name}' ({evaluation_design}) has "
                    f"{total_undefined[int(mm)]}/{occurrence_total} unresolved samples "
                    f"({undefined_frac * 100.0:.1f}%) could not find {mm} SO and {mm} OS neighbor(s)."
                )

            tail = compute_tail_metrics(sample_values, alpha=alpha)
            by_m[int(mm)] = CCMRResult(
                dataset=dataset_name,
                m=int(mm),
                value=value,
                std=std,
                n_pairs=len(pair_medians[int(mm)]),
                pair_values=finite_pair,
                sample_values=sample_values,
                sample_values_aligned=occurrence_values,
                occurrence_defined_mask=occurrence_defined_mask,
                undefined_frac=undefined_frac,
                evaluation_design=str(evaluation_design),
                evaluation_unit=str(evaluation_unit),
                occurrence_subsets=occurrence_subsets,
                occurrence_source_indices=occurrence_sources,
                k_start=int(k_start_value),
                k_final=int(k_final_value),
                retries=int(retries_value),
                alpha=tail.alpha,
                q_alpha=tail.q_alpha,
                ltm_alpha=tail.ltm_alpha,
            )

        ordered = {int(v): by_m[int(v)] for v in ordered_m_values}
        if return_single:
            return ordered[int(ordered_m_values[0])]
        return ordered

    @classmethod
    def _infer_dataset_name(cls, df: pd.DataFrame) -> str:
        if "dataset" not in df.columns or len(df) == 0:
            return "dataset"
        return str(df["dataset"].iloc[0]).strip() or "dataset"
