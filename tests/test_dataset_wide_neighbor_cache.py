"""Parity: the shared dataset-wide neighbour cache must reproduce the direct path exactly.

The refactor lets dataset-wide kNN curves, RI/MaRI scoring and k-selection share a single
prepared neighbour cache instead of each re-preparing neighbours. These tests pin that the
prepared-cache path is numerically identical to the pre-refactor direct path.
"""

import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI
from croma.metrics.neighbors import _knn_balanced_accuracy_by_k


def _dataset() -> tuple[np.ndarray, pd.DataFrame]:
    """24 samples, 2 labels x 3 confounders, two feature clusters with confounder offsets."""
    rng = np.random.default_rng(7)
    rows = []
    feats = []
    conf_offset = {"V1": 0.0, "V2": 0.18, "V3": 0.36}
    for i in range(24):
        label = "A" if i % 2 == 0 else "B"
        conf = ["V1", "V2", "V3"][i % 3]
        base = np.array([1.0, 0.0, 0.0]) if label == "A" else np.array([0.0, 1.0, 0.0])
        vec = base + np.array([0.0, 0.0, conf_offset[conf]]) + 0.02 * rng.standard_normal(3)
        feats.append(vec)
        rows.append(
            {
                "sample_id": f"s{i}",
                "image_path": f"/tmp/{i}.png",
                "label": label,
                "scanner_vendor": conf,
                "group_id": f"slide-{i}",
                "dataset": "toy",
            }
        )
    return np.asarray(feats, dtype=float), pd.DataFrame(rows)


K_VALUES = [2, 3, 4, 5]
SELECTED_K = 3
CONF = "scanner_vendor"


def _direct_artifacts(metric, **kwargs):
    return metric._compute_artifacts(
        features=FEATURES,
        manifest=MANIFEST,
        confounder_column=CONF,
        k_values=K_VALUES,
        evaluation_design="dataset_wide",
        selected_k=SELECTED_K,
        include_selected_result=True,
        **kwargs,
    )


def _cached_artifacts(metric, **kwargs):
    df = metric._normalize_manifest_inputs(MANIFEST, confounder_column=CONF)
    cache = metric._prepare_dataset_wide_neighbor_cache(features=FEATURES, df=df, k_values=K_VALUES)
    return metric._compute_artifacts_from_prepared_dataset_wide(
        prepared_neighbors=cache,
        dataset_name=metric._infer_dataset_name(df),
        k_values=K_VALUES,
        selected_k=SELECTED_K,
        include_selected_result=True,
        **kwargs,
    )


FEATURES, MANIFEST = _dataset()


def _assert_artifacts_equal(direct, cached) -> None:
    assert set(direct.curve) == set(cached.curve)
    for k in direct.curve:
        assert cached.curve[k] == pytest.approx(direct.curve[k], abs=0, rel=0)
    assert cached.result.value == pytest.approx(direct.result.value, abs=0, rel=0)
    assert cached.result.undefined_frac == pytest.approx(direct.result.undefined_frac, abs=0, rel=0)
    np.testing.assert_array_equal(
        np.asarray(cached.result.sample_values_aligned, dtype=float),
        np.asarray(direct.result.sample_values_aligned, dtype=float),
    )


def test_ri_dataset_wide_cache_matches_direct() -> None:
    _assert_artifacts_equal(_direct_artifacts(RI), _cached_artifacts(RI))


def test_mari_dataset_wide_cache_matches_direct() -> None:
    _assert_artifacts_equal(_direct_artifacts(MaRI, tau=0.2), _cached_artifacts(MaRI, tau=0.2))


def test_knn_curves_from_cache_match_direct() -> None:
    df = RI._normalize_manifest_inputs(MANIFEST, confounder_column=CONF)
    prepared_inputs = RI._prepare_dataset_wide_inputs(features=FEATURES, df=df)
    cache = RI._prepare_dataset_wide_neighbor_cache(features=FEATURES, df=df, k_values=K_VALUES)
    for target, encoded in (
        ("label", prepared_inputs.labels),
        ("confounder", prepared_inputs.centers),
    ):
        direct = _knn_balanced_accuracy_by_k(
            features=prepared_inputs.features,
            labels=encoded,
            group_ids=prepared_inputs.group_ids,
            k_values=K_VALUES,
            warn_context="parity",
        )
        cached = RI._knn_balanced_accuracy_by_k_from_prepared_subsets(
            prepared_subsets=[cache],
            target=target,
            k_values=K_VALUES,
            warn_context="parity",
        )
        assert set(direct) == set(cached)
        for k in direct:
            assert cached[k] == pytest.approx(direct[k], abs=0, rel=0)


def test_k_selection_from_cache_matches_direct() -> None:
    df = RI._normalize_manifest_inputs(MANIFEST, confounder_column=CONF)
    prepared_inputs = RI._prepare_dataset_wide_inputs(features=FEATURES, df=df)
    cache = RI._prepare_dataset_wide_neighbor_cache(features=FEATURES, df=df, k_values=K_VALUES)
    direct_k = RI._select_dataset_wide_k(
        prepared=prepared_inputs, k_candidates=K_VALUES, dataset_name="toy"
    )
    cached_k = RI._select_dataset_wide_k_from_prepared(
        prepared_neighbors=cache, k_candidates=K_VALUES, dataset_name="toy"
    )
    assert cached_k == direct_k
