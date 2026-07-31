import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from croma.metrics.base import (
    EVALUATION_DESIGN_ALL,
    EVALUATION_DESIGN_PAIRED_2X2,
    _normalize_evaluation_design,
)
from croma.metrics.neighbors import (
    _filter_query_neighbors_excluding_same_group,
    _initial_n_neighbors,
)
from croma.metrics.pairs import (
    GROUP_COLUMN,
    EvaluationSubset,
    ensure_canonical_manifest_columns,
    normalize_manifest,
    resolve_manifest_subsets,
    validate_subset_manifest,
)
from croma.metrics.tail import compute_tail_metrics
from croma.types import CRoMaResult

logger = logging.getLogger("croma")

# Headline per-type averaging radius. ``m=1`` is the single-neighbour estimate and
# is maximally sensitive to one outlier neighbour; ``m=5`` is the smallest window
# that removes single-neighbour dominance (no neighbour exceeds 20% of the estimate)
# while staying in the local typed shell. The pooled-median model ranking and the
# biology-/confounder-dominant sign are invariant across the m-sweep (Spearman
# >= 0.99), so m only affects per-sample magnitudes and tail statistics; m=1 and the
# full sweep are reported as sensitivity. See paper/sections/croma.tex.
CROMA_HEADLINE_M = 5


@dataclass(frozen=True)
class _CRoMaSearchMeta:
    k_start: int
    k_final: int
    retries: int


@dataclass(frozen=True)
class _SampleCRoMa:
    """Per-sample CRoMa together with *why* each undefined sample is undefined.

    A NaN in ``values`` has two possible causes, and the output cannot tell them apart:
    the search never found ``m`` neighbors of both types (``unresolved``), or it found
    them all at distance zero so the margin has no denominator (``zero_distance``). The
    two masks are disjoint and their union is exactly the NaN set, so a report built from
    them accounts for every undefined sample and attributes none of them by inference.
    """

    values: np.ndarray
    unresolved: np.ndarray
    zero_distance: np.ndarray


def _sample_croma_with_causes(
    so_dists: np.ndarray,
    os_dists: np.ndarray,
) -> _SampleCRoMa:
    """Per-sample CRoMa as a signed, normalized margin in ``(-1, 1)``, plus its NaN causes.

    ``CRoMa_i = (d_OS - d_SO) / (d_OS + d_SO)`` where ``d_SO``/``d_OS`` are the
    mean cosine distances to the ``m`` nearest ``SO``/``OS`` neighbors. The sign
    reports which typed neighbor is closer and the magnitude how decisively:
    ``> 0`` is biology-dominant (robust), ``< 0`` confounder-dominant (fragile),
    ``0`` an exactly contested boundary. Equivalently, the same-confounder
    impostor accounts for a fraction ``(1 + CRoMa_i) / 2`` of the total typed
    distance ``d_OS + d_SO`` and the biological match the remaining
    ``(1 - CRoMa_i) / 2``.

    An unfilled slot is ``inf``, which is the search's own record that it ran out of
    radius before filling it -- so ``unresolved`` reads the cause off the search rather
    than off the NaN it produces.
    """
    has_inf_so = np.any(np.isinf(so_dists), axis=1)
    has_inf_os = np.any(np.isinf(os_dists), axis=1)
    unresolved = has_inf_so | has_inf_os

    mean_so = np.mean(so_dists, axis=1)
    mean_os = np.mean(os_dists, axis=1)
    denom = mean_os + mean_so

    with np.errstate(divide="ignore", invalid="ignore"):
        croma = (mean_os - mean_so) / denom

    zero_distance = ~unresolved & (denom == 0.0)
    croma[unresolved] = np.nan
    croma[zero_distance] = np.nan
    return _SampleCRoMa(values=croma, unresolved=unresolved, zero_distance=zero_distance)


def _compute_sample_croma(
    so_dists: np.ndarray,
    os_dists: np.ndarray,
) -> np.ndarray:
    """Per-sample CRoMa alone; see :func:`_sample_croma_with_causes` for the definition."""
    return _sample_croma_with_causes(so_dists, os_dists).values


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
    group_ids: np.ndarray,
    m: int,
    start_k: int,
    k_growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, _CRoMaSearchMeta]:
    n_samples = int(len(labels))
    so_dists = np.full((n_samples, int(m)), np.inf, dtype=float)
    os_dists = np.full((n_samples, int(m)), np.inf, dtype=float)

    if n_samples <= 1:
        return (
            so_dists,
            os_dists,
            _CRoMaSearchMeta(
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
                _CRoMaSearchMeta(
                    k_start=k_start_used,
                    k_final=int(k_current),
                    retries=int(retries),
                ),
            )

        fetch_neighbors = _initial_n_neighbors(
            kmax=int(k_current), group_ids=group_ids, n_samples=n_samples
        )
        distances, raw_neighbors = model.kneighbors(
            features[query_indices], n_neighbors=fetch_neighbors
        )

        neigh_idx, neigh_dist, valid_counts = _filter_query_neighbors_excluding_same_group(
            raw_neighbors=raw_neighbors,
            raw_distances=distances,
            query_indices=query_indices,
            group_ids=group_ids,
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
                _CRoMaSearchMeta(
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


class CrossConfounderRobustnessMargin:
    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        manifest: pd.DataFrame,
        *,
        confounder_column: str,
        evaluation_design: str = EVALUATION_DESIGN_ALL,
        m: int | list[int] | tuple[int, ...] = CROMA_HEADLINE_M,
        alpha: float = 0.10,
        start_k: int = 200,
        k_growth_factor: float = 2.0,
    ) -> CRoMaResult | dict[int, CRoMaResult]:
        """Compute the cross-confounder robustness margin.

        Args:
            evaluation_design: ``"all"`` (the default) scores every supplied manifest row
                together, as one evaluation scope, at sample level. ``"paired_2x2"`` scores
                only the manifest's explicitly declared 2x2 subsets, at occurrence level.
        """
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

        df = normalize_manifest(manifest, confounder_column=confounder_column, source="manifest")
        ensure_canonical_manifest_columns(df, "manifest")
        dataset_name = cls._infer_dataset_name(df)
        if evaluation_design == EVALUATION_DESIGN_PAIRED_2X2:
            validate_subset_manifest(df, f"manifest for dataset '{dataset_name}'")

        if evaluation_design == EVALUATION_DESIGN_PAIRED_2X2:
            subsets = resolve_manifest_subsets(df)
            if not subsets:
                raise RuntimeError(
                    f"{dataset_name}: no valid manifest-defined 2x2 subsets remain for CRoMa"
                )
            evaluation_unit = "occurrence"
        else:
            subsets = [_dataset_subset(df)]
            evaluation_unit = "sample"

        pair_medians: dict[int, list[float]] = {int(mm): [] for mm in unique_m_values}
        occurrence_values_by_m: dict[int, list[np.ndarray]] = {
            int(mm): [] for mm in unique_m_values
        }
        occurrence_subsets_by_m: dict[int, list[np.ndarray]] = {
            int(mm): [] for mm in unique_m_values
        }
        occurrence_sources_by_m: dict[int, list[np.ndarray]] = {
            int(mm): [] for mm in unique_m_values
        }
        occurrence_total = 0
        total_undefined: dict[int, int] = {int(mm): 0 for mm in unique_m_values}
        # The undefined count, split by the cause each sample actually had. Kept apart from
        # ``total_undefined`` so the warning names a cause instead of inferring one.
        total_unresolved: dict[int, int] = {int(mm): 0 for mm in unique_m_values}
        total_zero_distance: dict[int, int] = {int(mm): 0 for mm in unique_m_values}

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
            subset_features = subset_features / (
                np.linalg.norm(subset_features, axis=1, keepdims=True) + 1e-12
            )

            labels = pd.factorize(sub["label"])[0].astype(int)
            centers = pd.factorize(sub["confounder"])[0].astype(int)
            group_ids = sub[GROUP_COLUMN].astype(str).to_numpy()

            so_dists, os_dists, search_meta = _iterative_typed_neighbor_search(
                features=subset_features,
                labels=labels,
                centers=centers,
                group_ids=group_ids,
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
                scored = _sample_croma_with_causes(so_dists[:, : int(mm)], os_dists[:, : int(mm)])
                sample_croma = scored.values
                informative = np.isfinite(sample_croma)
                n_informative = int(informative.sum())
                n_undefined = int(n_sub - n_informative)
                total_undefined[int(mm)] += n_undefined
                total_unresolved[int(mm)] += int(scored.unresolved.sum())
                total_zero_distance[int(mm)] += int(scored.zero_distance.sum())

                occurrence_values_by_m[int(mm)].append(np.asarray(sample_croma, dtype=float))
                occurrence_subsets_by_m[int(mm)].append(
                    np.full(n_sub, str(subset.subset_id), dtype=object)
                )
                occurrence_sources_by_m[int(mm)].append(idx.astype(int))

                if n_informative > 0:
                    pair_medians[int(mm)].append(float(np.median(sample_croma[informative])))
                else:
                    pair_medians[int(mm)].append(float("nan"))

        k_start_value = int(min(k_start_values)) if k_start_values else 0
        k_final_value = int(max(k_final_values)) if k_final_values else 0
        retries_value = int(max(retries_values)) if retries_values else 0

        by_m: dict[int, CRoMaResult] = {}
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
            undefined_frac = (
                float(total_undefined[int(mm)] / occurrence_total) if occurrence_total > 0 else 0.0
            )

            if total_undefined[int(mm)] > 0:
                causes: list[str] = []
                if total_unresolved[int(mm)] > 0:
                    causes.append(
                        f"{total_unresolved[int(mm)]} where the neighbour search could not "
                        f"find {mm} SO and {mm} OS neighbor(s) before reaching its radius cap"
                    )
                if total_zero_distance[int(mm)] > 0:
                    causes.append(
                        f"{total_zero_distance[int(mm)]} where the search did find them, all "
                        f"at distance 0, so the margin's denominator d_OS + d_SO is 0 -- what "
                        f"a collapsed embedding, or a manifest that duplicates rows, produces"
                    )
                logger.warning(
                    f"[CRoMa] dataset '{dataset_name}' ({evaluation_design}) leaves "
                    f"{total_undefined[int(mm)]}/{occurrence_total} samples "
                    f"({undefined_frac * 100.0:.1f}%) undefined at m={mm}: "
                    f"{'; '.join(causes)}."
                )

            tail = compute_tail_metrics(sample_values, alpha=alpha)
            by_m[int(mm)] = CRoMaResult(
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
