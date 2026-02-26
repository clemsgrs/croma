from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mari.metrics.neighbors import _prepare_neighbors

logger = logging.getLogger("mari")
from mari.metrics.tail import compute_tail_metrics
from mari.metrics.pairs import (
    ensure_required_columns,
    infer_2x2_pairs,
    normalize_center_values,
    subset_by_pair,
)
from mari.types import CCRRResult

_DEFAULT_KMAX = 200
_PAIR_MODES = {"paired", "global"}


def _find_typed_neighbor_distances(
    labels: np.ndarray,
    centers: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_dist: np.ndarray,
    valid_counts: np.ndarray,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(labels)
    so_dists = np.full((n_samples, m), np.inf, dtype=float)
    os_dists = np.full((n_samples, m), np.inf, dtype=float)

    for i in range(n_samples):
        eff_k = int(valid_counts[i])
        so_count = 0
        os_count = 0
        for pos in range(eff_k):
            j = int(neigh_idx[i, pos])
            if j < 0:
                continue
            d = float(neigh_dist[i, pos])
            same_label = labels[j] == labels[i]
            same_center = centers[j] == centers[i]
            if same_label and not same_center and so_count < m:
                so_dists[i, so_count] = d
                so_count += 1
            elif not same_label and same_center and os_count < m:
                os_dists[i, os_count] = d
                os_count += 1
            if so_count >= m and os_count >= m:
                break

    return so_dists, os_dists


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


class CrossConfounderRetrievalRatio:

    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        mode: str,
        m: int = 1,
        alpha: float = 0.10,
        exclude_centers: object | None = None,
        max_pairs: int | None = None,
        random_state: int = 0,
        kmax: int | None = None,
    ) -> CCRRResult:
        mode_value = str(mode).strip().lower()
        if mode_value not in _PAIR_MODES:
            raise ValueError("mode must be one of {'paired', 'global'}")
        if m < 1:
            raise ValueError("m must be >= 1")

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

        if mode_value == "paired":
            pairs = infer_2x2_pairs(
                df,
                dataset_name=dataset_name,
                max_pairs=max_pairs,
                random_state=random_state,
            )
            if not pairs:
                raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for CCRR")
            subsets = [subset_by_pair(df, pair) for pair in pairs]
        else:
            subsets = [df]

        effective_kmax = kmax if kmax is not None else min(len(df) - 1, _DEFAULT_KMAX)
        effective_kmax = max(1, min(effective_kmax, len(df) - 1))

        pair_medians: list[float] = []
        sample_sum = np.zeros(len(features), dtype=float)
        sample_count = np.zeros(len(features), dtype=int)
        total_samples = 0
        total_undefined = 0

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

            sub_kmax = max(1, min(effective_kmax, len(sub) - 1))
            neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(
                pair_features, slide_ids, sub_kmax
            )

            so_dists, os_dists = _find_typed_neighbor_distances(
                labels, centers, neigh_idx, neigh_dist, valid_counts, m
            )
            sample_ccrr = _compute_sample_ccrr(so_dists, os_dists)

            informative = np.isfinite(sample_ccrr)
            n_informative = int(informative.sum())
            n_sub = len(sub)
            total_samples += n_sub
            total_undefined += n_sub - n_informative

            if n_informative > 0:
                pair_medians.append(float(np.median(sample_ccrr[informative])))
                global_idx = idx[informative]
                sample_sum[global_idx] += sample_ccrr[informative]
                sample_count[global_idx] += 1
            else:
                pair_medians.append(float("nan"))

        finite_pair = np.asarray(pair_medians, dtype=float)
        finite_mask = np.isfinite(finite_pair)
        if finite_mask.any():
            value = float(np.median(finite_pair[finite_mask]))
            std = float(finite_pair[finite_mask].std(ddof=0)) if finite_mask.sum() > 1 else 0.0
        else:
            value = float("nan")
            std = 0.0

        has_sample = sample_count > 0
        if has_sample.any():
            sample_values = (sample_sum[has_sample] / sample_count[has_sample]).astype(float)
        else:
            sample_values = np.empty((0,), dtype=float)

        undefined_frac = float(total_undefined / total_samples) if total_samples > 0 else 0.0

        if total_undefined > 0:
            logger.warning(
                f"[CCRR] {total_undefined}/{total_samples} samples "
                f"({undefined_frac * 100.0:.1f}%) could not find {m} SO and {m} OS "
                f"neighbor(s) within kmax={effective_kmax}. "
                f"Consider increasing kmax to reduce undefined samples."
            )

        tail = compute_tail_metrics(sample_values, alpha=alpha)

        return CCRRResult(
            dataset=dataset_name,
            m=m,
            value=value,
            std=std,
            n_pairs=len(pair_medians),
            pair_values=finite_pair,
            sample_values=sample_values,
            undefined_frac=undefined_frac,
            alpha=tail.alpha,
            q_alpha=tail.q_alpha,
            ltm_alpha=tail.ltm_alpha,
        )

    @classmethod
    def _infer_dataset_name(cls, df: pd.DataFrame) -> str:
        if "dataset" not in df.columns or len(df) == 0:
            return "dataset"
        return str(df["dataset"].iloc[0]).strip() or "dataset"
