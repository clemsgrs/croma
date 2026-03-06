import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_results as ar


def _toy_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "A",
                "ri": 0.90,
                "mari": 0.85,
                "ccrr": 1.30,
                "ccrr_q_alpha": 0.82,
                "ccrr_ltm_alpha": 0.70,
                "bio_knn_bacc": 0.97,
                "center_knn_bacc": 0.55,
                "ri_undefined_frac": 0.10,
                "mari_undefined_frac": 0.10,
                "ccrr_undefined_frac": 0.00,
                "ccrr_retries": 8,
                "ccrr_k_final": 5000,
            },
            {
                "model": "B",
                "ri": 0.80,
                "mari": 0.88,
                "ccrr": 1.10,
                "ccrr_q_alpha": 0.75,
                "ccrr_ltm_alpha": 0.65,
                "bio_knn_bacc": 0.96,
                "center_knn_bacc": 0.60,
                "ri_undefined_frac": 0.12,
                "mari_undefined_frac": 0.12,
                "ccrr_undefined_frac": 0.00,
                "ccrr_retries": 9,
                "ccrr_k_final": 6000,
            },
            {
                "model": "C",
                "ri": 0.70,
                "mari": 0.74,
                "ccrr": 0.95,
                "ccrr_q_alpha": 0.60,
                "ccrr_ltm_alpha": 0.52,
                "bio_knn_bacc": 0.95,
                "center_knn_bacc": 0.66,
                "ri_undefined_frac": 0.20,
                "mari_undefined_frac": 0.20,
                "ccrr_undefined_frac": 0.05,
                "ccrr_retries": 12,
                "ccrr_k_final": 9000,
            },
            {
                "model": "D",
                "ri": 0.60,
                "mari": 0.62,
                "ccrr": 0.80,
                "ccrr_q_alpha": 0.50,
                "ccrr_ltm_alpha": 0.45,
                "bio_knn_bacc": 0.93,
                "center_knn_bacc": 0.72,
                "ri_undefined_frac": 0.30,
                "mari_undefined_frac": 0.30,
                "ccrr_undefined_frac": 0.10,
                "ccrr_retries": 15,
                "ccrr_k_final": 12000,
            },
        ]
    )


def _toy_metrics_with_rank_and_tail_risks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "A",
                "ri": 0.95,
                "mari": 0.92,
                "ccrr": 0.40,
                "ccrr_q_alpha": 0.30,
                "ccrr_ltm_alpha": 0.25,
                "bio_knn_bacc": 0.97,
                "center_knn_bacc": 0.56,
                "ri_undefined_frac": 0.80,
                "mari_undefined_frac": 0.78,
                "ccrr_undefined_frac": 0.00,
                "ccrr_retries": 11,
                "ccrr_k_final": 11000,
            },
            {
                "model": "B",
                "ri": 0.85,
                "mari": 0.84,
                "ccrr": 0.70,
                "ccrr_q_alpha": 0.66,
                "ccrr_ltm_alpha": 0.62,
                "bio_knn_bacc": 0.96,
                "center_knn_bacc": 0.61,
                "ri_undefined_frac": 0.20,
                "mari_undefined_frac": 0.22,
                "ccrr_undefined_frac": 0.00,
                "ccrr_retries": 7,
                "ccrr_k_final": 5500,
            },
            {
                "model": "C",
                "ri": 0.75,
                "mari": 0.73,
                "ccrr": 0.90,
                "ccrr_q_alpha": 0.55,
                "ccrr_ltm_alpha": 0.48,
                "bio_knn_bacc": 0.94,
                "center_knn_bacc": 0.66,
                "ri_undefined_frac": 0.18,
                "mari_undefined_frac": 0.18,
                "ccrr_undefined_frac": 0.00,
                "ccrr_retries": 10,
                "ccrr_k_final": 9800,
            },
        ]
    )


def _toy_phase2_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.80,
                "mari": 0.82,
                "ccrr": 1.10,
                "ccrr_m": 1,
                "ccrr_q_alpha": 0.20,
                "ccrr_ltm_alpha": 0.11,
                "bio_knn_bacc": 0.95,
                "center_knn_bacc": 0.60,
                "ri_undefined_frac": 0.0,
                "mari_undefined_frac": 0.0,
                "ccrr_undefined_frac": 0.0,
            }
        ]
    )


def _toy_phase2_per_sample() -> pd.DataFrame:
    rows: list[dict] = []
    values = [
        ("A", "C1", 0.10),
        ("A", "C1", 0.11),
        ("A", "C1", 0.12),
        ("A", "C2", 0.40),
        ("A", "C2", 0.41),
        ("A", "C2", 0.42),
        ("B", "C1", 0.50),
        ("B", "C1", 0.51),
        ("B", "C1", 0.52),
        ("B", "C2", 0.60),
        ("B", "C2", 0.61),
        ("B", "C2", 0.62),
    ]
    for idx, (label, center, ccrr) in enumerate(values):
        rows.append(
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "sample_index": idx,
                "sample_id": f"s{idx}",
                "slide_id": f"slide-{idx}",
                "label": label,
                "medical_center": center,
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.5,
                "mari": 0.5,
                "ri_defined": True,
                "mari_defined": True,
                "ri_undefined_type": 0,
                "mari_undefined_type": 0,
                "ccrr_m1": ccrr,
            }
        )
    return pd.DataFrame(rows)


def _toy_phase3_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    for model in ("A", "B", "C", "D"):
        rows.append(
            {
                "dataset": "toy",
                "model": model,
                "mode": "global",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.80,
                "mari": 0.82,
                "ccrr": 1.10,
                "ccrr_m": 1,
                "ccrr_q_alpha": 0.20,
                "ccrr_ltm_alpha": 0.11,
                "bio_knn_bacc": 0.95,
                "center_knn_bacc": 0.60,
                "ri_undefined_frac": 0.0,
                "mari_undefined_frac": 0.0,
                "ccrr_undefined_frac": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _toy_phase3_per_sample() -> pd.DataFrame:
    values_by_model = {
        "A": [0.10, 0.11, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],  # tail {s0,s1}
        "B": [0.40, 0.10, 0.11, 0.60, 0.70, 0.80, 0.90, 1.00],  # tail {s1,s2}
        "C": [0.40, 0.50, 0.60, 0.70, 0.10, 0.11, 0.90, 1.00],  # tail {s4,s5}
        "D": [0.10, 0.11, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],  # tail {s0,s1}
    }
    rows: list[dict] = []
    for model, values in values_by_model.items():
        for idx, ccrr in enumerate(values):
            label = "A" if idx < 4 else "B"
            center = "C1" if idx % 2 == 0 else "C2"
            rows.append(
                {
                    "dataset": "toy",
                    "model": model,
                    "mode": "global",
                    "sample_index": idx,
                    "sample_id": f"s{idx}",
                    "slide_id": f"slide-{idx}",
                    "label": label,
                    "medical_center": center,
                    "k": 3,
                    "tau": 0.2,
                    "ccrr_alpha": 0.25,
                    "ccrr_search": "start=200;growth=2;alpha=0.25",
                    "excluded_centers": "",
                    "ri": 0.5,
                    "mari": 0.5,
                    "ri_defined": True,
                    "mari_defined": True,
                    "ri_undefined_type": 0,
                    "mari_undefined_type": 0,
                    "ccrr_m1": ccrr,
                }
            )
    return pd.DataFrame(rows)


def _toy_phase4_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.80,
                "mari": 0.82,
                "ccrr": 1.10,
                "ccrr_m": 1,
                "ccrr_q_alpha": 0.20,
                "ccrr_ltm_alpha": 0.11,
                "bio_knn_bacc": 0.95,
                "center_knn_bacc": 0.60,
                "ri_undefined_frac": 0.0,
                "mari_undefined_frac": 0.0,
                "ccrr_undefined_frac": 0.0,
            }
        ]
    )


def _toy_phase4_per_sample() -> pd.DataFrame:
    rows: list[dict] = []
    for idx in range(12):
        if idx < 4:
            slide_id = "slide-hot"
            ccrr = [0.10, 0.11, 0.12, 0.13][idx]
        elif idx < 8:
            slide_id = "slide-mid-a"
            ccrr = [0.50, 0.51, 0.52, 0.53][idx - 4]
        else:
            slide_id = "slide-mid-b"
            ccrr = [0.60, 0.61, 0.62, 0.63][idx - 8]
        rows.append(
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "sample_index": idx,
                "sample_id": f"s{idx}",
                "slide_id": slide_id,
                "label": "A" if idx < 6 else "B",
                "medical_center": "C1" if idx % 2 == 0 else "C2",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.5,
                "mari": 0.5,
                "ri_defined": True,
                "mari_defined": True,
                "ri_undefined_type": 0,
                "mari_undefined_type": 0,
                "ccrr_m1": ccrr,
            }
        )
    return pd.DataFrame(rows)


def test_analyze_results_writes_expected_outputs_and_ri_ranks(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_metrics().to_csv(metrics_path, index=False)
    out_dir = tmp_path / "analysis"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--metrics-csv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--rank-reference",
            "RI",
            "--no-plots",
        ],
    )
    code = ar.main()
    assert code == 0

    expected_files = [
        out_dir / "correlation_pearson.csv",
        out_dir / "correlation_spearman.csv",
        out_dir / "model_ranks.csv",
        out_dir / "top_models_by_metric.csv",
        out_dir / "rank_deltas.csv",
        out_dir / "rank_agreement.csv",
        out_dir / "model_action_flags.csv",
        out_dir / "analysis_report.md",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing output file: {path}"

    ranks = pd.read_csv(out_dir / "model_ranks.csv")
    assert "rank_ri" in ranks.columns
    assert set(c for c in ranks.columns if c.startswith("rank_")) == {"rank_ri", "rank_mari", "rank_ccrr"}
    best_ri_model = ranks.sort_values("rank_ri", ascending=True).iloc[0]["model"]
    assert best_ri_model == "A"

    pearson = pd.read_csv(out_dir / "correlation_pearson.csv", index_col=0)
    assert set(pearson.columns) == {"ri", "mari", "ccrr"}
    assert set(pearson.index) == {"ri", "mari", "ccrr"}

    deltas = pd.read_csv(out_dir / "rank_deltas.csv")
    assert set(deltas["pair"]) == {"ri_vs_mari", "ri_vs_ccrr", "mari_vs_ccrr"}
    assert set(deltas["direction"]).issubset({"improvement", "downgrade", "no_change"})
    assert "improvement_delta" in deltas.columns
    assert "abs_improvement_delta" in deltas.columns

    top = pd.read_csv(out_dir / "top_models_by_metric.csv")
    assert set(top["metric"]) == {"ri", "mari", "ccrr", "ccrr_q_alpha", "ccrr_ltm_alpha"}
    top_ri = top[top["metric"] == "ri"].sort_values("rank_position").iloc[0]["model"]
    assert top_ri == "A"

    report = (out_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert "Additional Insights and Action Flags" in report
    assert "No rank shifts with |delta| >= 2." in report


def test_analyze_results_writes_ccrr_stratum_enrichment_outputs(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_phase2_metrics().to_csv(metrics_path, index=False)
    _toy_phase2_per_sample().to_csv(tmp_path / "per_sample_metrics.csv", index=False)
    out_dir = tmp_path / "analysis"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--metrics-csv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--no-plots",
        ],
    )
    assert ar.main() == 0

    enrichment_path = out_dir / "ccrr_stratum_enrichment.csv"
    summary_path = out_dir / "ccrr_stratum_concentration.csv"
    heatmap_path = out_dir / "ccrr_stratum_enrichment_heatmap.csv"
    assert enrichment_path.exists()
    assert summary_path.exists()
    assert heatmap_path.exists()

    enrichment_df = pd.read_csv(enrichment_path)
    summary_df = pd.read_csv(summary_path)
    flagged = enrichment_df.loc[
        (enrichment_df["label"] == "A") & (enrichment_df["medical_center"] == "C1")
    ].iloc[0]
    assert flagged["dataset_count"] == 3
    assert flagged["dataset_frac"] == 0.25
    assert flagged["tail_count"] == 3
    assert flagged["tail_frac"] == 1.0
    assert flagged["enrichment_ratio"] == 4.0
    assert flagged["tail_mean_ccrr"] == 0.11
    assert flagged["tail_median_ccrr"] == 0.11
    assert bool(flagged["flagged"])

    summary_row = summary_df.iloc[0]
    assert summary_row["model"] == "A"
    assert summary_row["effective_tail_strata"] == 1.0
    assert summary_row["tail_sample_count"] == 3

    report = (out_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert "CCRR Tail Stratum Enrichment" in report


def test_ccrr_stratum_enrichment_requires_minimum_tail_support() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.20,
                "ccrr_search": "start=200;growth=2;alpha=0.2",
                "excluded_centers": "",
                "ccrr_m": 1,
            }
        ]
    )
    per_sample_df = pd.DataFrame(
        [
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s0", "slide_id": "slide-0", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s1", "slide_id": "slide-1", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s2", "slide_id": "slide-2", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.50},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s3", "slide_id": "slide-3", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.60},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s4", "slide_id": "slide-4", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.61},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s5", "slide_id": "slide-5", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.62},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s6", "slide_id": "slide-6", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.63},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s7", "slide_id": "slide-7", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.64},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s8", "slide_id": "slide-8", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.65},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s9", "slide_id": "slide-9", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.66},
        ]
    )

    enrichment_df, summary_df, heatmap_df = ar._ccrr_stratum_enrichment(metrics_df, per_sample_df)

    target = enrichment_df.loc[
        (enrichment_df["label"] == "A") & (enrichment_df["medical_center"] == "C1")
    ].iloc[0]
    assert target["tail_count"] == 2
    assert target["enrichment_ratio"] > 2.0
    assert not bool(target["flagged"])
    assert len(summary_df) == 1
    assert len(heatmap_df) == len(enrichment_df)


def test_tail_support_threshold_scales_with_tail_size() -> None:
    assert ar._tail_support_threshold(1) == 3
    assert ar._tail_support_threshold(60) == 3
    assert ar._tail_support_threshold(61) == 4
    assert ar._tail_support_threshold(100) == 5


def test_ccrr_stratum_concentration_captures_diffuse_tail() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.5,
                "ccrr_search": "start=200;growth=2;alpha=0.5",
                "excluded_centers": "",
                "ccrr_m": 1,
            }
        ]
    )
    per_sample_df = pd.DataFrame(
        [
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s0", "slide_id": "slide-0", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s1", "slide_id": "slide-1", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s2", "slide_id": "slide-2", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.12},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s3", "slide_id": "slide-3", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.13},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s4", "slide_id": "slide-4", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.70},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s5", "slide_id": "slide-5", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.71},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s6", "slide_id": "slide-6", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.72},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s7", "slide_id": "slide-7", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.73},
        ]
    )

    _, summary_df, _ = ar._ccrr_stratum_enrichment(metrics_df, per_sample_df)

    summary_row = summary_df.iloc[0]
    assert summary_row["tail_sample_count"] == 4
    assert summary_row["tail_strata_count"] == 4
    assert summary_row["effective_tail_strata"] == 4.0


def test_analyze_results_writes_ccrr_tail_overlap_outputs(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_phase3_metrics().to_csv(metrics_path, index=False)
    _toy_phase3_per_sample().to_csv(tmp_path / "per_sample_metrics.csv", index=False)
    out_dir = tmp_path / "analysis"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--metrics-csv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--no-plots",
        ],
    )
    assert ar.main() == 0

    jaccard_path = out_dir / "ccrr_tail_overlap_jaccard.csv"
    summary_path = out_dir / "ccrr_tail_overlap_summary.csv"
    always_path = out_dir / "ccrr_tail_overlap_always_fragile_samples.csv"
    unique_path = out_dir / "ccrr_tail_overlap_unique_fragile_samples.csv"
    assert jaccard_path.exists()
    assert summary_path.exists()
    assert always_path.exists()
    assert unique_path.exists()

    jaccard = pd.read_csv(jaccard_path)
    assert set(jaccard["model_a"]) == {"A", "B", "C", "D"}
    assert set(jaccard["model_b"]) == {"A", "B", "C", "D"}
    assert jaccard.loc[(jaccard["model_a"] == "A") & (jaccard["model_b"] == "A"), "jaccard"].iloc[0] == 1.0
    assert jaccard.loc[(jaccard["model_a"] == "A") & (jaccard["model_b"] == "B"), "jaccard"].iloc[0] == 1.0 / 3.0
    assert jaccard.loc[(jaccard["model_a"] == "A") & (jaccard["model_b"] == "C"), "jaccard"].iloc[0] == 0.0
    assert jaccard.loc[(jaccard["model_a"] == "A") & (jaccard["model_b"] == "D"), "jaccard"].iloc[0] == 1.0
    assert jaccard.loc[(jaccard["model_a"] == "B") & (jaccard["model_b"] == "C"), "jaccard"].iloc[0] == 0.0

    summary_df = pd.read_csv(summary_path)
    summary_a = summary_df.loc[summary_df["model"] == "A"].iloc[0]
    summary_b = summary_df.loc[summary_df["model"] == "B"].iloc[0]
    summary_c = summary_df.loc[summary_df["model"] == "C"].iloc[0]
    summary_d = summary_df.loc[summary_df["model"] == "D"].iloc[0]
    assert summary_a["always_fragile_count"] == 0
    assert summary_a["unique_fragile_count"] == 0
    assert summary_b["unique_fragile_count"] == 1
    assert summary_c["unique_fragile_count"] == 2
    assert summary_d["unique_fragile_count"] == 0

    always_df = pd.read_csv(always_path)
    unique_df = pd.read_csv(unique_path)
    assert len(always_df) == 0
    assert sorted(unique_df.loc[unique_df["model"] == "B", "sample_id"].tolist()) == ["s2"]
    assert sorted(unique_df.loc[unique_df["model"] == "C", "sample_id"].tolist()) == ["s4", "s5"]

    report = (out_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert "CCRR Tail Overlap" in report


def test_ccrr_tail_overlap_requires_identical_sample_universe() -> None:
    metrics_df = _toy_phase3_metrics().iloc[:2].copy()
    per_sample_df = _toy_phase3_per_sample()
    per_sample_df = per_sample_df.loc[
        ~((per_sample_df["model"] == "B") & (per_sample_df["sample_id"] == "s7"))
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match="same sample universe"):
        ar._ccrr_tail_overlap(metrics_df, per_sample_df)


def test_ccrr_tail_overlap_groups_multiple_contexts() -> None:
    metrics_df = pd.concat(
        [
            _toy_phase3_metrics(),
            _toy_phase3_metrics().assign(ccrr_alpha=0.5, ccrr_search="start=200;growth=2;alpha=0.5"),
        ],
        ignore_index=True,
    )
    per_sample_df = pd.concat(
        [
            _toy_phase3_per_sample(),
            _toy_phase3_per_sample().assign(ccrr_alpha=0.5, ccrr_search="start=200;growth=2;alpha=0.5"),
        ],
        ignore_index=True,
    )

    jaccard_df, summary_df, always_df, unique_df = ar._ccrr_tail_overlap(metrics_df, per_sample_df)

    assert set(jaccard_df["ccrr_alpha"]) == {0.25, 0.5}
    assert len(summary_df) == 8
    assert set(summary_df["ccrr_alpha"]) == {0.25, 0.5}
    assert set(always_df["ccrr_alpha"]) == {0.5}
    assert set(unique_df["ccrr_alpha"]) == {0.25, 0.5}


def test_slide_support_threshold_scales_with_tail_size() -> None:
    assert ar._slide_support_threshold(1) == 3
    assert ar._slide_support_threshold(30) == 3
    assert ar._slide_support_threshold(31) == 4
    assert ar._slide_support_threshold(100) == 10


def test_analyze_results_writes_ccrr_slide_diagnostics_outputs(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_phase4_metrics().to_csv(metrics_path, index=False)
    _toy_phase4_per_sample().to_csv(tmp_path / "per_sample_metrics.csv", index=False)
    out_dir = tmp_path / "analysis"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--metrics-csv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--no-plots",
        ],
    )
    assert ar.main() == 0

    enrichment_path = out_dir / "ccrr_slide_enrichment.csv"
    summary_path = out_dir / "ccrr_slide_summary.csv"
    assert enrichment_path.exists()
    assert summary_path.exists()

    enrichment_df = pd.read_csv(enrichment_path)
    summary_df = pd.read_csv(summary_path)
    flagged = enrichment_df.loc[enrichment_df["slide_id"] == "slide-hot"].iloc[0]
    assert flagged["dataset_count"] == 4
    assert flagged["dataset_frac"] == pytest.approx(1.0 / 3.0)
    assert flagged["tail_count"] == 3
    assert flagged["tail_frac"] == 1.0
    assert flagged["enrichment_ratio"] == pytest.approx(3.0)
    assert flagged["tail_mean_ccrr"] == pytest.approx(0.11)
    assert flagged["tail_median_ccrr"] == pytest.approx(0.11)
    assert bool(flagged["flagged"])

    summary_row = summary_df.iloc[0]
    assert summary_row["flagged_slides_count"] == 1
    assert summary_row["flagged_tail_mass_frac"] == 1.0

    report = (out_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert "CCRR Slide Diagnostics" in report


def test_ccrr_slide_diagnostics_require_stricter_support_than_strata() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "mode": "global",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.20,
                "ccrr_search": "start=200;growth=2;alpha=0.2",
                "excluded_centers": "",
                "ccrr_m": 1,
            }
        ]
    )
    per_sample_df = pd.DataFrame(
        [
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s0", "slide_id": "slide-hot", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s1", "slide_id": "slide-hot", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s2", "slide_id": "slide-other-a", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.50},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s3", "slide_id": "slide-other-a", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.51},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s4", "slide_id": "slide-other-b", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.60},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s5", "slide_id": "slide-other-b", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.61},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s6", "slide_id": "slide-other-c", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.62},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s7", "slide_id": "slide-other-c", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.63},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s8", "slide_id": "slide-other-d", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.64},
            {"dataset": "toy", "model": "A", "mode": "global", "sample_id": "s9", "slide_id": "slide-other-d", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.65},
        ]
    )

    enrichment_df, summary_df = ar._ccrr_slide_diagnostics(metrics_df, per_sample_df)

    target = enrichment_df.loc[enrichment_df["slide_id"] == "slide-hot"].iloc[0]
    assert target["tail_count"] == 2
    assert target["enrichment_ratio"] > 3.0
    assert not bool(target["flagged"])
    assert summary_df.iloc[0]["flagged_slides_count"] == 0


def test_analyze_results_adds_sweep_insights_and_flags(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_metrics_with_rank_and_tail_risks().to_csv(metrics_path, index=False)
    out_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {"model": "A", "k": 5, "ri": 0.95, "mari": 0.91},
            {"model": "A", "k": 25, "ri": 0.70, "mari": 0.63},
            {"model": "B", "k": 5, "ri": 0.85, "mari": 0.84},
            {"model": "B", "k": 25, "ri": 0.80, "mari": 0.79},
            {"model": "C", "k": 5, "ri": 0.75, "mari": 0.73},
            {"model": "C", "k": 25, "ri": 0.71, "mari": 0.69},
        ]
    ).to_csv(tmp_path / "k_sweep_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": "A", "m": 1, "ccrr": 0.40, "ccrr_q_alpha": 0.30, "ccrr_ltm_alpha": 0.25, "ccrr_retries": 11, "ccrr_k_final": 11000},
            {"model": "A", "m": 10, "ccrr": 0.58, "ccrr_q_alpha": 0.41, "ccrr_ltm_alpha": 0.35, "ccrr_retries": 6, "ccrr_k_final": 7000},
            {"model": "B", "m": 1, "ccrr": 0.70, "ccrr_q_alpha": 0.66, "ccrr_ltm_alpha": 0.62, "ccrr_retries": 5, "ccrr_k_final": 4500},
            {"model": "B", "m": 10, "ccrr": 0.74, "ccrr_q_alpha": 0.69, "ccrr_ltm_alpha": 0.66, "ccrr_retries": 4, "ccrr_k_final": 4300},
            {"model": "C", "m": 1, "ccrr": 0.90, "ccrr_q_alpha": 0.55, "ccrr_ltm_alpha": 0.48, "ccrr_retries": 10, "ccrr_k_final": 9800},
            {"model": "C", "m": 10, "ccrr": 0.93, "ccrr_q_alpha": 0.61, "ccrr_ltm_alpha": 0.57, "ccrr_retries": 6, "ccrr_k_final": 6400},
        ]
    ).to_csv(tmp_path / "ccrr_m_sweep_metrics.csv", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--metrics-csv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--rank-reference",
            "RI",
            "--no-plots",
        ],
    )
    code = ar.main()
    assert code == 0

    k_sensitivity_path = out_dir / "k_sweep_sensitivity.csv"
    ccrr_sensitivity_path = out_dir / "ccrr_m_sweep_sensitivity.csv"
    assert k_sensitivity_path.exists()
    assert ccrr_sensitivity_path.exists()

    flags = pd.read_csv(out_dir / "model_action_flags.csv")
    assert "flag" in flags.columns
    assert (flags["flag"] == "rank_shift_ri_vs_ccrr").any()
    assert (flags["flag"] == "coverage_risk_ri").any()
    assert (flags["flag"] == "tail_gap_ltm_high").any()
    assert (flags["flag"] == "k_sweep_sensitivity_high").any()
    assert (flags["flag"] == "ccrr_search_cost_high").any()

    report = (out_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert "K-Sweep Sensitivity" in report
    assert "CCRR m-Sweep Sensitivity and Cost" in report
