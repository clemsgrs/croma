
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from mari.metrics.neighbors import _filter_query_neighbors_excluding_same_slide, _initial_n_neighbors

logger = logging.getLogger("mari")
from mari.metrics.tail import compute_tail_metrics
from mari.metrics.pairs import (
    ensure_required_columns,
    infer_2x2_pairs,
    normalize_center_values,
    subset_by_pair,
)
from mari.types import CCRRResult


@dataclass(frozen=True)
class _CCRRSearchMeta:
    acceptance_met: bool
    k_start: int
    k_final: int
    retries: int


def _compute_sample_ccrr(
    so_dists: np.ndarray,
    os_dists: np.ndarray,
) -> np.ndarray:
    has_inf_so = np.any(np.isinf(so_dists), axis=1)
    has_inf_os = np.any(np.isinf(os_dists), axis=1)
    undefined = has_inf_so | has_inf_os

    mean_so = np.mean(so_dists, axis=1)
    mean_os = np.mean(os_dists, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ccrr = mean_os / mean_so

    ccrr[undefined] = np.nan
    ccrr[mean_so == 0.0] = np.nan
    return ccrr


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
    acceptance_threshold: float,
    start_k: int,
    k_growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, _CCRRSearchMeta]:
    n_samples = int(len(labels))
    so_dists = np.full((n_samples, int(m)), np.inf, dtype=float)
    os_dists = np.full((n_samples, int(m)), np.inf, dtype=float)

    if n_samples <= 1:
        return (
            so_dists,
            os_dists,
            _CCRRSearchMeta(
                acceptance_met=True,
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
                _CCRRSearchMeta(
                    acceptance_met=True,
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

        undefined_frac = float(np.count_nonzero(unresolved_mask)) / float(n_samples)
        if undefined_frac <= float(acceptance_threshold):
            return (
                so_dists,
                os_dists,
                _CCRRSearchMeta(
                    acceptance_met=True,
                    k_start=k_start_used,
                    k_final=int(k_current),
                    retries=int(retries),
                ),
            )

        if int(k_current) >= n_samples - 1:
            return (
                so_dists,
                os_dists,
                _CCRRSearchMeta(
                    acceptance_met=False,
                    k_start=k_start_used,
                    k_final=int(k_current),
                    retries=int(retries),
                ),
            )

        retries += 1
        grown_k = int(math.ceil(float(k_current) * float(k_growth_factor)))
        k_current = int(min(n_samples - 1, max(int(k_current) + 1, grown_k)))


class CrossConfounderRetrievalRatio:

    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        m: int | list[int] | tuple[int, ...] = 1,
        alpha: float = 0.10,
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
        acceptance_threshold: float = 0.0,
        start_k: int = 200,
        k_growth_factor: float = 2.0,
    ) -> CCRRResult | dict[int, CCRRResult]:
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
        if float(acceptance_threshold) < 0.0 or float(acceptance_threshold) > 1.0:
            raise ValueError("acceptance_threshold must be in [0, 1]")
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

        excluded = normalize_center_values(exclude_centers)
        if excluded:
            center_series = df["medical_center"].map(lambda v: str(v).strip())
            keep_mask = ~center_series.isin(excluded)
            if not bool(keep_mask.any()):
                excluded_txt = ", ".join(excluded)
                raise ValueError(
                    f"No samples remain after excluding centers [{excluded_txt}] from dataset '{dataset_name}'"
                )
            features = features[keep_mask.to_numpy()]
            df = df.loc[keep_mask].reset_index(drop=True)

        pairs = infer_2x2_pairs(
            df,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            random_state=random_state,
        )
        if not pairs:
            raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for CCRR")
        subsets = [subset_by_pair(df, pair) for pair in pairs]

        pair_medians: dict[int, list[float]] = {int(m): [] for m in unique_m_values}
        sample_sum: dict[int, np.ndarray] = {
            int(m): np.zeros(len(features), dtype=float) for m in unique_m_values
        }
        sample_count: dict[int, np.ndarray] = {
            int(m): np.zeros(len(features), dtype=int) for m in unique_m_values
        }
        total_samples = 0
        total_undefined: dict[int, int] = {int(m): 0 for m in unique_m_values}

        k_start_values: list[int] = []
        k_final_values: list[int] = []
        retries_values: list[int] = []
        acceptance_met_values: dict[int, list[bool]] = {int(m): [] for m in unique_m_values}
        m_max = int(max(unique_m_values))

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

            so_dists, os_dists, search_meta = _iterative_typed_neighbor_search(
                features=pair_features,
                labels=labels,
                centers=centers,
                slide_ids=slide_ids,
                m=int(m_max),
                acceptance_threshold=float(acceptance_threshold),
                start_k=int(start_k),
                k_growth_factor=float(k_growth_factor),
            )

            n_sub = len(sub)
            total_samples += n_sub

            k_start_values.append(int(search_meta.k_start))
            k_final_values.append(int(search_meta.k_final))
            retries_values.append(int(search_meta.retries))
            for m in unique_m_values:
                sample_ccrr = _compute_sample_ccrr(so_dists[:, : int(m)], os_dists[:, : int(m)])
                informative = np.isfinite(sample_ccrr)
                n_informative = int(informative.sum())
                n_undefined = int(n_sub - n_informative)
                total_undefined[int(m)] += n_undefined
                if int(m) == int(m_max):
                    acceptance_met_values[int(m)].append(bool(search_meta.acceptance_met))
                else:
                    acceptance_met_values[int(m)].append(
                        bool((float(n_undefined) / float(n_sub)) <= float(acceptance_threshold))
                    )

                if n_informative > 0:
                    pair_medians[int(m)].append(float(np.median(sample_ccrr[informative])))
                    global_idx = idx[informative]
                    sample_sum[int(m)][global_idx] += sample_ccrr[informative]
                    sample_count[int(m)][global_idx] += 1
                else:
                    pair_medians[int(m)].append(float("nan"))

        k_start_value = int(min(k_start_values)) if k_start_values else 0
        k_final_value = int(max(k_final_values)) if k_final_values else 0
        retries_value = int(max(retries_values)) if retries_values else 0

        by_m: dict[int, CCRRResult] = {}
        for m in unique_m_values:
            finite_pair = np.asarray(pair_medians[int(m)], dtype=float)
            finite_mask = np.isfinite(finite_pair)
            if finite_mask.any():
                value = float(np.median(finite_pair[finite_mask]))
                std = float(finite_pair[finite_mask].std(ddof=0)) if finite_mask.sum() > 1 else 0.0
            else:
                value = float("nan")
                std = 0.0

            has_sample = sample_count[int(m)] > 0
            sample_values_aligned = np.full((len(features),), np.nan, dtype=float)
            if has_sample.any():
                sample_values_aligned[has_sample] = (
                    sample_sum[int(m)][has_sample] / sample_count[int(m)][has_sample]
                ).astype(float)
                sample_values = sample_values_aligned[has_sample].astype(float)
            else:
                sample_values = np.empty((0,), dtype=float)

            undefined_frac = float(total_undefined[int(m)] / total_samples) if total_samples > 0 else 0.0
            acceptance_met = bool(all(acceptance_met_values[int(m)])) if acceptance_met_values[int(m)] else True

            if total_samples > 0 and not acceptance_met:
                logger.warning(
                    f"[CCRR] undefined threshold unmet: {total_undefined[int(m)]}/{total_samples} "
                    f"({undefined_frac * 100.0:.1f}%) > target {float(acceptance_threshold) * 100.0:.1f}% "
                    f"after reaching k={k_final_value}. Returning best-effort result."
                )

            if total_undefined[int(m)] > 0:
                logger.warning(
                    f"[CCRR] {total_undefined[int(m)]}/{total_samples} samples "
                    f"({undefined_frac * 100.0:.1f}%) could not find {m} SO and {m} OS neighbor(s)."
                )

            tail = compute_tail_metrics(sample_values, alpha=alpha)
            by_m[int(m)] = CCRRResult(
                dataset=dataset_name,
                m=int(m),
                value=value,
                std=std,
                n_pairs=len(pair_medians[int(m)]),
                pair_values=finite_pair,
                sample_values=sample_values,
                sample_values_aligned=sample_values_aligned,
                undefined_frac=undefined_frac,
                acceptance_threshold=float(acceptance_threshold),
                acceptance_met=bool(acceptance_met),
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
