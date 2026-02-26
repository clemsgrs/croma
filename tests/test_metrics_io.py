from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import metrics_io as mio


def test_k_candidates_signature_is_sorted_and_unique() -> None:
    assert mio.k_candidates_signature([7, 3, 7, 5]) == "3,5,7"


def test_excluded_centers_signature_is_sorted_and_unique() -> None:
    assert mio.excluded_centers_signature([" C2 ", "C1", "C2"]) == "C1,C2"


def test_ccrr_search_signature_is_stable() -> None:
    sig = mio.ccrr_search_signature(
        acceptance_threshold=0.0,
        start_k=200,
        k_growth_factor=1.5,
    )
    assert sig == "thr=0;start=200;growth=1.5"


def test_load_cached_rows_requires_matching_k_candidates_signature(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "3,5,7",
                "ri": 0.5,
            }
        ]
    ).to_csv(metrics_csv, index=False)

    hit = mio.load_cached_rows(
        metrics_csv=metrics_csv,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="3,5,7",
    )
    assert "A" in hit

    miss = mio.load_cached_rows(
        metrics_csv=metrics_csv,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="3,5",
    )
    assert miss == {}


def test_load_cached_rows_requires_matching_excluded_centers_signature(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "3,5,7",
                "excluded_centers": "C3",
                "ri": 0.5,
            }
        ]
    ).to_csv(metrics_csv, index=False)

    hit = mio.load_cached_rows(
        metrics_csv=metrics_csv,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="3,5,7",
        excluded_centers_sig="C3",
    )
    assert "A" in hit

    miss = mio.load_cached_rows(
        metrics_csv=metrics_csv,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="3,5,7",
        excluded_centers_sig="",
    )
    assert miss == {}


def test_load_cached_rows_requires_matching_ccrr_search_signature(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "3,5,7",
                "excluded_centers": "none",
                "ccrr_search": "thr=0;start=200;growth=1.5",
                "ri": 0.5,
            }
        ]
    ).to_csv(metrics_csv, index=False)

    hit = mio.load_cached_rows(
        metrics_csv=metrics_csv,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="3,5,7",
        excluded_centers_sig="none",
        ccrr_search_sig="thr=0;start=200;growth=1.5",
    )
    assert "A" in hit

    miss = mio.load_cached_rows(
        metrics_csv=metrics_csv,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="3,5,7",
        excluded_centers_sig="none",
        ccrr_search_sig="thr=0.25;start=200;growth=1.5",
    )
    assert miss == {}


def test_save_and_load_k_sweep_rows_with_compatibility(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "toy",
            "model": "A",
            "mode": "global",
            "tau": 0.2,
            "k_candidates": "1,3",
            "k": 1,
            "knn_bacc": 0.70,
            "ri": 0.40,
            "selected_k": 3,
            "embedding_path": "/tmp/a.npy",
        },
        {
            "dataset": "toy",
            "model": "A",
            "mode": "global",
            "tau": 0.2,
            "k_candidates": "1,3",
            "k": 3,
            "knn_bacc": 0.80,
            "ri": 0.60,
            "selected_k": 3,
            "embedding_path": "/tmp/a.npy",
        },
    ]
    csv_path = tmp_path / "k_sweep.csv"
    json_path = tmp_path / "k_sweep.json"

    mio.save_k_sweep_metrics(rows=rows, csv_path=csv_path, json_path=json_path)
    assert csv_path.exists()
    assert json_path.exists()

    cached = mio.load_cached_k_sweep_rows(
        metrics_csv=csv_path,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="1,3",
        expected_k_values=[1, 3],
    )
    assert list(cached) == ["A"]
    assert [int(r["k"]) for r in cached["A"]] == [1, 3]

    miss = mio.load_cached_k_sweep_rows(
        metrics_csv=csv_path,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="1,5",
        expected_k_values=[1, 5],
    )
    assert miss == {}


def test_load_cached_k_sweep_rows_rejects_partial_k_set(tmp_path: Path) -> None:
    csv_path = tmp_path / "k_sweep.csv"
    pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "1,3",
                "k": 1,
                "knn_bacc": 0.70,
                "ri": 0.40,
                "selected_k": 1,
                "embedding_path": "/tmp/a.npy",
            }
        ]
    ).to_csv(csv_path, index=False)

    miss = mio.load_cached_k_sweep_rows(
        metrics_csv=csv_path,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="1,3",
        expected_k_values=[1, 3],
    )
    assert miss == {}


def test_load_cached_k_sweep_rows_rejects_duplicate_k_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "k_sweep.csv"
    pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "1,3",
                "k": 1,
                "knn_bacc": 0.70,
                "ri": 0.40,
                "selected_k": 1,
                "embedding_path": "/tmp/a.npy",
            },
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "1,3",
                "k": 1,
                "knn_bacc": 0.71,
                "ri": 0.41,
                "selected_k": 1,
                "embedding_path": "/tmp/a.npy",
            },
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "tau": 0.2,
                "k_candidates": "1,3",
                "k": 3,
                "knn_bacc": 0.80,
                "ri": 0.60,
                "selected_k": 3,
                "embedding_path": "/tmp/a.npy",
            },
        ]
    ).to_csv(csv_path, index=False)

    miss = mio.load_cached_k_sweep_rows(
        metrics_csv=csv_path,
        models=["A"],
        mode="global",
        tau=0.2,
        k_candidates_sig="1,3",
        expected_k_values=[1, 3],
    )
    assert miss == {}
