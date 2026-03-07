import sys
from pathlib import Path
import subprocess

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
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
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
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
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
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
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
                    "evaluation_design": "paired_2x2",
                    "evaluation_unit": "occurrence",
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
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
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
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
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


def _toy_trusted_cross_model_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
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
            },
            {
                "dataset": "toy",
                "model": "B",
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.79,
                "mari": 0.81,
                "ccrr": 1.05,
                "ccrr_m": 1,
                "ccrr_q_alpha": 0.21,
                "ccrr_ltm_alpha": 0.12,
                "bio_knn_bacc": 0.94,
                "center_knn_bacc": 0.61,
                "ri_undefined_frac": 0.0,
                "mari_undefined_frac": 0.0,
                "ccrr_undefined_frac": 0.0,
            },
            {
                "dataset": "toy",
                "model": "C",
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ri": 0.50,
                "mari": 0.52,
                "ccrr": 0.80,
                "ccrr_m": 1,
                "ccrr_q_alpha": 0.10,
                "ccrr_ltm_alpha": 0.09,
                "bio_knn_bacc": 0.80,
                "center_knn_bacc": 0.70,
                "ri_undefined_frac": 0.0,
                "mari_undefined_frac": 0.0,
                "ccrr_undefined_frac": 0.0,
            },
        ]
    )


def _toy_trusted_cross_model_per_sample() -> pd.DataFrame:
    rows: list[dict] = []
    values_by_model = {
        "A": [0.10, 0.11, 0.12, 0.13, 0.50, 0.51, 0.52, 0.53, 0.60, 0.61, 0.62, 0.63],
        "B": [0.10, 0.11, 0.12, 0.40, 0.50, 0.51, 0.52, 0.53, 0.60, 0.61, 0.62, 0.63],
        "C": [0.50, 0.51, 0.52, 0.53, 0.60, 0.61, 0.62, 0.63, 0.10, 0.11, 0.12, 0.13],
    }
    for model, values in values_by_model.items():
        for idx, ccrr in enumerate(values):
            if idx < 4:
                slide_id = "slide-hot"
                label = "A"
                center = "C1"
            elif idx < 6:
                slide_id = "slide-mid-a"
                label = "A"
                center = "C2"
            else:
                slide_id = "slide-mid-b"
                label = "B"
                center = "C2"
            rows.append(
                {
                    "dataset": "toy",
                    "model": model,
                    "evaluation_design": "paired_2x2",
                    "evaluation_unit": "occurrence",
                    "sample_index": idx,
                    "sample_id": f"s{idx}",
                    "slide_id": slide_id,
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


def test_match_per_sample_rows_respects_evaluation_design() -> None:
    metric_row = pd.Series(
        {
            "dataset": "toy",
            "model": "A",
            "evaluation_design": "paired_2x2",
            "evaluation_unit": "occurrence",
            "k": 3,
            "tau": 0.2,
            "ccrr_alpha": 0.25,
            "ccrr_search": "start=200;growth=2;alpha=0.25",
            "excluded_centers": "",
        }
    )
    per_sample_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
                "evaluation_design": "paired_2x2",
                "evaluation_unit": "occurrence",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "sample_id": "paired",
            },
            {
                "dataset": "toy",
                "model": "A",
                "evaluation_design": "dataset_wide",
                "evaluation_unit": "sample",
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "sample_id": "dataset",
            },
        ]
    )

    matched = ar._match_per_sample_rows(per_sample_df, metric_row)

    assert matched["sample_id"].tolist() == ["paired"]


def test_analyze_results_default_writes_question_focused_outputs(monkeypatch, tmp_path: Path) -> None:
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
        ],
    )
    code = ar.main()
    assert code == 0

    written = sorted(path.name for path in out_dir.iterdir())
    assert written == [
        "cross_model_findings.csv",
        "single_model_tail_characterization.csv",
        "single_model_tail_characterization.md",
    ]

    findings = pd.read_csv(out_dir / "single_model_tail_characterization.csv")
    assert set(findings["model"]) == {"A", "B", "C", "D"}
    best_ri_model = findings.loc[findings["question_scope"] == "tail_concentration"].sort_values("rank_ri").iloc[0]["model"]
    assert best_ri_model == "A"

    report = (out_dir / "single_model_tail_characterization.md").read_text(encoding="utf-8")
    assert "# Single-Model Tail Characterization" in report
    assert "## Overview" in report
    assert "## Model A" in report
    assert "### Tail Concentration" in report


def test_analyze_results_detailed_writes_retained_appendix_outputs(monkeypatch, tmp_path: Path) -> None:
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
            "--detailed",
        ],
    )
    assert ar.main() == 0

    assert (out_dir / "single_model_tail_characterization.md").exists()
    assert (out_dir / "single_model_tail_characterization.csv").exists()
    assert (out_dir / "cross_model_findings.csv").exists()

    enrichment_path = out_dir / "ccrr_stratum_enrichment.csv"
    summary_path = out_dir / "ccrr_stratum_concentration.csv"
    flagged_slide_conc_path = out_dir / "ccrr_flagged_stratum_slide_concentration.csv"
    assert enrichment_path.exists()
    assert summary_path.exists()
    assert flagged_slide_conc_path.exists()
    assert not (out_dir / "correlation_pearson.csv").exists()
    assert not (out_dir / "correlation_spearman.csv").exists()
    assert not (out_dir / "top_models_by_metric.csv").exists()
    assert not (out_dir / "rank_deltas.csv").exists()
    assert not (out_dir / "rank_agreement.csv").exists()
    assert not (out_dir / "ccrr_stratum_enrichment_heatmap.csv").exists()

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

    label_enrichment_df = pd.read_csv(out_dir / "ccrr_label_enrichment.csv")
    label_summary_df = pd.read_csv(out_dir / "ccrr_label_summary.csv")
    center_enrichment_df = pd.read_csv(out_dir / "ccrr_center_enrichment.csv")
    center_summary_df = pd.read_csv(out_dir / "ccrr_center_summary.csv")
    flagged_slide_conc_df = pd.read_csv(flagged_slide_conc_path)

    label_row = label_enrichment_df.loc[label_enrichment_df["label"] == "A"].iloc[0]
    assert label_row["dataset_count"] == 6
    assert label_row["dataset_frac"] == 0.5
    assert label_row["tail_count"] == 3
    assert label_row["tail_frac"] == 1.0
    assert label_row["enrichment_ratio"] == 2.0
    assert label_row["tail_mean_ccrr"] == 0.11
    assert bool(label_row["flagged"])

    center_row = center_enrichment_df.loc[center_enrichment_df["medical_center"] == "C1"].iloc[0]
    assert center_row["dataset_count"] == 6
    assert center_row["dataset_frac"] == 0.5
    assert center_row["tail_count"] == 3
    assert center_row["tail_frac"] == 1.0
    assert center_row["enrichment_ratio"] == 2.0
    assert center_row["tail_mean_ccrr"] == 0.11
    assert bool(center_row["flagged"])

    assert label_summary_df.iloc[0]["flagged_label_count"] == 1
    assert center_summary_df.iloc[0]["flagged_center_count"] == 1

    flagged_stratum_slide_row = flagged_slide_conc_df.loc[
        (flagged_slide_conc_df["label"] == "A")
        & (flagged_slide_conc_df["medical_center"] == "C1")
        & (flagged_slide_conc_df["slide_id"] == "slide-0")
    ].iloc[0]
    assert flagged_stratum_slide_row["slide_tail_count"] == 1
    assert flagged_stratum_slide_row["stratum_tail_count"] == 3
    assert flagged_stratum_slide_row["tail_frac_within_stratum"] == pytest.approx(1.0 / 3.0)
    assert flagged_stratum_slide_row["stratum_frac"] == pytest.approx(1.0 / 3.0)
    assert flagged_stratum_slide_row["stratum_enrichment_ratio"] == pytest.approx(1.0)
    assert not bool(flagged_stratum_slide_row["flagged"])

    report = (out_dir / "single_model_tail_characterization.md").read_text(encoding="utf-8")
    assert "### Stratum Enrichment" in report
    assert "A / C1" in report
    assert "### Center Enrichment" in report
    assert "C1" in report


def test_flagged_stratum_slide_concentration_captures_within_stratum_localization() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
               
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ccrr_m": 1,
            }
        ]
    )
    per_sample_df = pd.DataFrame(
        [
            {"dataset": "toy", "model": "A", "sample_id": "s0", "slide_id": "slide-hot", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "sample_id": "s1", "slide_id": "slide-hot", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "sample_id": "s2", "slide_id": "slide-warm", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.12},
            {"dataset": "toy", "model": "A", "sample_id": "s3", "slide_id": "slide-cool", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.40},
            {"dataset": "toy", "model": "A", "sample_id": "s4", "slide_id": "slide-cool", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.41},
            {"dataset": "toy", "model": "A", "sample_id": "s5", "slide_id": "slide-cool", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.42},
            {"dataset": "toy", "model": "A", "sample_id": "s6", "slide_id": "slide-other", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.50},
            {"dataset": "toy", "model": "A", "sample_id": "s7", "slide_id": "slide-other", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.51},
            {"dataset": "toy", "model": "A", "sample_id": "s8", "slide_id": "slide-other", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.60},
            {"dataset": "toy", "model": "A", "sample_id": "s9", "slide_id": "slide-other", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.61},
            {"dataset": "toy", "model": "A", "sample_id": "s10", "slide_id": "slide-other", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.62},
            {"dataset": "toy", "model": "A", "sample_id": "s11", "slide_id": "slide-other", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.63},
        ]
    )

    stratum_enrichment_df, _, _ = ar._ccrr_stratum_enrichment(metrics_df, per_sample_df)
    concentration_df = ar._ccrr_flagged_stratum_slide_concentration(
        metrics_df,
        per_sample_df,
        stratum_enrichment_df,
    )

    rows = concentration_df.loc[
        (concentration_df["label"] == "A") & (concentration_df["medical_center"] == "C1")
    ].sort_values("slide_id")
    assert rows["slide_id"].tolist() == ["slide-hot", "slide-warm"]
    hot = rows.loc[rows["slide_id"] == "slide-hot"].iloc[0]
    assert hot["slide_tail_count"] == 2
    assert hot["stratum_tail_count"] == 3
    assert hot["slide_stratum_count"] == 2
    assert hot["tail_frac_within_stratum"] == pytest.approx(2.0 / 3.0)
    assert hot["stratum_frac"] == pytest.approx(2.0 / 3.0)
    assert hot["stratum_enrichment_ratio"] == pytest.approx(1.0)
    assert not bool(hot["flagged"])

    warm = rows.loc[rows["slide_id"] == "slide-warm"].iloc[0]
    assert warm["slide_tail_count"] == 1
    assert warm["stratum_tail_count"] == 3
    assert warm["slide_stratum_count"] == 1
    assert warm["tail_frac_within_stratum"] == pytest.approx(1.0 / 3.0)
    assert warm["stratum_frac"] == pytest.approx(1.0 / 3.0)
    assert warm["stratum_enrichment_ratio"] == pytest.approx(1.0)
    assert not bool(warm["flagged"])


def test_flagged_stratum_slide_concentration_computes_within_stratum_baseline_enrichment() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
               
                "k": 3,
                "tau": 0.2,
                "ccrr_alpha": 0.25,
                "ccrr_search": "start=200;growth=2;alpha=0.25",
                "excluded_centers": "",
                "ccrr_m": 1,
            }
        ]
    )
    per_sample_df = pd.DataFrame(
        [
            {"dataset": "toy", "model": "A", "sample_id": "s0", "slide_id": "slide-hot", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "sample_id": "s1", "slide_id": "slide-hot", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "sample_id": "s2", "slide_id": "slide-warm", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.12},
            {"dataset": "toy", "model": "A", "sample_id": "s3", "slide_id": "slide-cool-a", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.60},
            {"dataset": "toy", "model": "A", "sample_id": "s4", "slide_id": "slide-cool-b", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.61},
            {"dataset": "toy", "model": "A", "sample_id": "s5", "slide_id": "slide-cool-c", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.62},
            {"dataset": "toy", "model": "A", "sample_id": "s6", "slide_id": "slide-other-a", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.70},
            {"dataset": "toy", "model": "A", "sample_id": "s7", "slide_id": "slide-other-b", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.71},
            {"dataset": "toy", "model": "A", "sample_id": "s8", "slide_id": "slide-other-c", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.72},
            {"dataset": "toy", "model": "A", "sample_id": "s9", "slide_id": "slide-other-d", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.73},
            {"dataset": "toy", "model": "A", "sample_id": "s10", "slide_id": "slide-other-e", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.74},
            {"dataset": "toy", "model": "A", "sample_id": "s11", "slide_id": "slide-other-f", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.25, "ccrr_search": "start=200;growth=2;alpha=0.25", "excluded_centers": "", "ccrr_m1": 0.75},
        ]
    )

    stratum_enrichment_df, _, _ = ar._ccrr_stratum_enrichment(metrics_df, per_sample_df)
    concentration_df = ar._ccrr_flagged_stratum_slide_concentration(
        metrics_df,
        per_sample_df,
        stratum_enrichment_df,
    )

    rows = concentration_df.loc[
        (concentration_df["label"] == "A") & (concentration_df["medical_center"] == "C1")
    ].sort_values("slide_id")
    assert rows["slide_id"].tolist() == ["slide-hot", "slide-warm"]
    hot = rows.loc[rows["slide_id"] == "slide-hot"].iloc[0]
    assert hot["slide_tail_count"] == 2
    assert hot["stratum_tail_count"] == 3
    assert hot["slide_stratum_count"] == 2
    assert hot["tail_frac_within_stratum"] == pytest.approx(2.0 / 3.0)
    assert hot["stratum_frac"] == pytest.approx(2.0 / 6.0)
    assert hot["stratum_enrichment_ratio"] == pytest.approx(2.0)
    assert bool(hot["flagged"])

    warm = rows.loc[rows["slide_id"] == "slide-warm"].iloc[0]
    assert warm["slide_tail_count"] == 1
    assert warm["slide_stratum_count"] == 1
    assert warm["tail_frac_within_stratum"] == pytest.approx(1.0 / 3.0)
    assert warm["stratum_frac"] == pytest.approx(1.0 / 6.0)
    assert warm["stratum_enrichment_ratio"] == pytest.approx(2.0)
    assert not bool(warm["flagged"])


def test_ccrr_stratum_enrichment_requires_minimum_tail_support() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
               
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
            {"dataset": "toy", "model": "A", "sample_id": "s0", "slide_id": "slide-0", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "sample_id": "s1", "slide_id": "slide-1", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "sample_id": "s2", "slide_id": "slide-2", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.50},
            {"dataset": "toy", "model": "A", "sample_id": "s3", "slide_id": "slide-3", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.60},
            {"dataset": "toy", "model": "A", "sample_id": "s4", "slide_id": "slide-4", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.61},
            {"dataset": "toy", "model": "A", "sample_id": "s5", "slide_id": "slide-5", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.62},
            {"dataset": "toy", "model": "A", "sample_id": "s6", "slide_id": "slide-6", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.63},
            {"dataset": "toy", "model": "A", "sample_id": "s7", "slide_id": "slide-7", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.64},
            {"dataset": "toy", "model": "A", "sample_id": "s8", "slide_id": "slide-8", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.65},
            {"dataset": "toy", "model": "A", "sample_id": "s9", "slide_id": "slide-9", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.66},
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
            {"dataset": "toy", "model": "A", "sample_id": "s0", "slide_id": "slide-0", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "sample_id": "s1", "slide_id": "slide-1", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "sample_id": "s2", "slide_id": "slide-2", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.12},
            {"dataset": "toy", "model": "A", "sample_id": "s3", "slide_id": "slide-3", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.13},
            {"dataset": "toy", "model": "A", "sample_id": "s4", "slide_id": "slide-4", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.70},
            {"dataset": "toy", "model": "A", "sample_id": "s5", "slide_id": "slide-5", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.71},
            {"dataset": "toy", "model": "A", "sample_id": "s6", "slide_id": "slide-6", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.72},
            {"dataset": "toy", "model": "A", "sample_id": "s7", "slide_id": "slide-7", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.5, "ccrr_search": "start=200;growth=2;alpha=0.5", "excluded_centers": "", "ccrr_m1": 0.73},
        ]
    )

    _, summary_df, _ = ar._ccrr_stratum_enrichment(metrics_df, per_sample_df)

    summary_row = summary_df.iloc[0]
    assert summary_row["tail_sample_count"] == 4
    assert summary_row["tail_strata_count"] == 4
    assert summary_row["effective_tail_strata"] == 4.0


def test_analyze_results_compact_cross_model_findings_summarize_shared_fragility(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_trusted_cross_model_metrics().to_csv(metrics_path, index=False)
    _toy_trusted_cross_model_per_sample().to_csv(tmp_path / "per_sample_metrics.csv", index=False)
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
        ],
    )
    assert ar.main() == 0

    findings = pd.read_csv(out_dir / "cross_model_findings.csv")
    assert set(findings["finding_scope"]) == {"label", "slide", "stratum"}

    top_label = findings.loc[findings["finding_scope"] == "label"].sort_values("rank_within_scope").iloc[0]
    assert top_label["label"] == "A"
    assert top_label["n_models_trusted"] == 2
    assert top_label["n_models_flagged"] == 2
    assert top_label["frac_models_flagged"] == 1.0

    top_slide = findings.loc[findings["finding_scope"] == "slide"].sort_values("rank_within_scope").iloc[0]
    assert top_slide["slide_id"] == "slide-hot"
    assert top_slide["n_models_trusted"] == 2
    assert top_slide["n_models_flagged"] == 2
    assert top_slide["frac_models_flagged"] == 1.0

    top_stratum = findings.loc[findings["finding_scope"] == "stratum"].sort_values("rank_within_scope").iloc[0]
    assert top_stratum["label"] == "A"
    assert top_stratum["medical_center"] == "C1"
    assert top_stratum["n_models_trusted"] == 2
    assert top_stratum["n_models_flagged"] == 2
    assert top_stratum["frac_models_flagged"] == 1.0

    assert "flagged_models" in findings.columns


def test_write_single_model_tail_characterization_renders_all_reported_pockets(tmp_path: Path) -> None:
    report_path = tmp_path / "single_model_tail_characterization.md"
    df_raw = pd.DataFrame([{"dataset": "toy"}])
    findings_df = pd.DataFrame(
        [
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "reliability",
                "flag": "coverage_risk_mari",
                "summary_text": "MaRI coverage is poor",
                "report_rank_within_model": 1,
            },
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "tail_concentration",
                "tail_sample_count": 6,
                "tail_strata_count": 2,
                "flagged_strata_count": 2,
                "effective_tail_strata": 1.8,
                "report_rank_within_model": 1,
            },
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "stratum_enrichment",
                "label": "tumor",
                "medical_center": "C1",
                "tail_count": 4,
                "tail_frac": 0.40,
                "dataset_count": 4,
                "dataset_frac": 0.20,
                "enrichment_ratio": 2.0,
                "tail_mean_ccrr": 0.20,
                "tail_median_ccrr": 0.21,
                "report_rank_within_model": 1,
            },
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "stratum_enrichment",
                "label": "normal",
                "medical_center": "C2",
                "tail_count": 3,
                "tail_frac": 0.30,
                "dataset_count": 3,
                "dataset_frac": 0.15,
                "enrichment_ratio": 2.0,
                "tail_mean_ccrr": 0.35,
                "tail_median_ccrr": 0.36,
                "report_rank_within_model": 2,
            },
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "stratum_severity",
                "label": "stroma",
                "medical_center": "C3",
                "tail_count": 3,
                "enrichment_ratio": 1.2,
                "tail_mean_ccrr": 0.08,
                "tail_median_ccrr": 0.08,
                "report_rank_within_model": 1,
            },
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "slide_enrichment",
                "slide_id": "slide-hot",
                "tail_count": 3,
                "tail_frac": 0.30,
                "dataset_count": 3,
                "dataset_frac": 0.08,
                "enrichment_ratio": 3.75,
                "tail_mean_ccrr": 0.10,
                "tail_median_ccrr": 0.10,
                "report_rank_within_model": 1,
            },
            {
                "model": "A",
                "rank_ri": 1.0,
                "rank_mari": 1.0,
                "rank_ccrr": 2.0,
                "question_scope": "stratum_slide_enrichment",
                "label": "tumor",
                "medical_center": "C1",
                "slide_id": "slide-hot",
                "slide_tail_count": 2,
                "slide_stratum_count": 2,
                "tail_frac_within_stratum": 0.50,
                "stratum_frac": 0.20,
                "stratum_enrichment_ratio": 2.5,
                "report_rank_within_model": 1,
            },
        ]
    )

    ar._write_single_model_tail_characterization_report(
        out_path=report_path,
        input_csv=tmp_path / "metrics.csv",
        df_raw=df_raw,
        characterization_df=findings_df,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Single-Model Tail Characterization" in report
    assert "## Model A" in report
    assert "### Stratum Enrichment" in report
    assert "tumor / C1" in report
    assert "normal / C2" in report
    assert "### Stratum Severity" in report
    assert "stroma / C3" in report
    assert "### Tail Concentration" in report
    assert "effective_tail_strata=1.80" in report
    assert "### Slide-Level Patterns" in report
    assert "slide-hot" in report


def test_ccrr_cross_model_prevalence_summarizes_shared_fragility() -> None:
    sample_df, stratum_df, label_df, slide_df = ar._ccrr_cross_model_prevalence(
        _toy_phase3_metrics(),
        _toy_phase3_per_sample(),
    )

    assert len(sample_df) == 8
    s0 = sample_df.loc[sample_df["sample_id"] == "s0"].iloc[0]
    s1 = sample_df.loc[sample_df["sample_id"] == "s1"].iloc[0]
    s4 = sample_df.loc[sample_df["sample_id"] == "s4"].iloc[0]
    assert s0["n_models_in_tail"] == 2
    assert s0["frac_models_in_tail"] == 0.5
    assert bool(s0["half_or_more_models_fragile"])
    assert s1["n_models_in_tail"] == 3
    assert s1["frac_models_in_tail"] == 0.75
    assert s4["n_models_in_tail"] == 1
    assert s4["frac_models_in_tail"] == 0.25

    label_a = label_df.loc[label_df["label"] == "A"].iloc[0]
    label_b = label_df.loc[label_df["label"] == "B"].iloc[0]
    assert label_a["mean_frac_models_in_tail"] == pytest.approx(0.375)
    assert label_a["n_half_or_more_models_fragile"] == 2
    assert label_b["mean_frac_models_in_tail"] == pytest.approx(0.125)
    assert label_b["n_half_or_more_models_fragile"] == 0

    stratum_ac1 = stratum_df.loc[
        (stratum_df["label"] == "A") & (stratum_df["medical_center"] == "C1")
    ].iloc[0]
    assert stratum_ac1["mean_frac_models_in_tail"] == pytest.approx(0.375)
    assert stratum_ac1["n_half_or_more_models_fragile"] == 1

    slide_s1 = slide_df.loc[slide_df["slide_id"] == "slide-1"].iloc[0]
    assert slide_s1["sample_count"] == 1
    assert slide_s1["mean_frac_models_in_tail"] == 0.75


def test_build_cross_model_findings_uses_trusted_models_and_flagged_agreement() -> None:
    metrics_df = _toy_trusted_cross_model_metrics()
    per_sample_df = _toy_trusted_cross_model_per_sample()
    stratum_df, _ = ar._ccrr_stratum_enrichment(metrics_df, per_sample_df)[:2]
    label_df, _ = ar._ccrr_label_enrichment(metrics_df, per_sample_df)
    slide_df, _ = ar._ccrr_slide_diagnostics(metrics_df, per_sample_df)

    findings = ar._build_cross_model_findings(
        df_metrics_raw=metrics_df,
        ccrr_label_enrichment_df=label_df,
        ccrr_stratum_enrichment_df=stratum_df,
        ccrr_slide_enrichment_df=slide_df,
    )

    assert set(findings["finding_scope"]) == {"label", "slide", "stratum"}
    label_top = findings.loc[findings["finding_scope"] == "label"].sort_values("rank_within_scope").iloc[0]
    assert label_top["label"] == "A"
    assert label_top["n_models_trusted"] == 2
    assert label_top["n_models_flagged"] == 2
    assert label_top["frac_models_flagged"] == 1.0
    assert label_top["median_enrichment_ratio"] == pytest.approx(2.0)

    slide_top = findings.loc[findings["finding_scope"] == "slide"].sort_values("rank_within_scope").iloc[0]
    assert slide_top["slide_id"] == "slide-hot"
    assert slide_top["n_models_trusted"] == 2
    assert slide_top["n_models_flagged"] == 2
    assert slide_top["frac_models_flagged"] == 1.0

    assert "C" not in findings.get("flagged_models", pd.Series(dtype=str)).astype(str).str.cat(sep=",")


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


def test_build_single_model_tail_characterization_reports_all_flagged_and_supported_severe_pockets() -> None:
    metrics_df = _toy_phase2_metrics()
    rank_df = ar._rank_table(metrics_df[["model", "ri", "mari", "ccrr"]], metrics=["ri", "mari", "ccrr"])
    action_flags_df = pd.DataFrame(
        [
            {"model": "A", "flag": "coverage_risk_mari", "severity": "high", "detail": "MaRI undefined fraction is 60%."},
        ]
    )
    stratum_enrichment_df = pd.DataFrame(
        [
            {"model": "A", "label": "A", "medical_center": "C1", "tail_count": 4, "tail_support_threshold": 3, "tail_frac": 0.40, "dataset_count": 4, "dataset_frac": 0.20, "enrichment_ratio": 2.0, "tail_mean_ccrr": 0.20, "tail_median_ccrr": 0.21, "flagged": True},
            {"model": "A", "label": "B", "medical_center": "C2", "tail_count": 3, "tail_support_threshold": 3, "tail_frac": 0.30, "dataset_count": 3, "dataset_frac": 0.15, "enrichment_ratio": 2.0, "tail_mean_ccrr": 0.35, "tail_median_ccrr": 0.36, "flagged": True},
            {"model": "A", "label": "C", "medical_center": "C3", "tail_count": 3, "tail_support_threshold": 3, "tail_frac": 0.10, "dataset_count": 6, "dataset_frac": 0.30, "enrichment_ratio": 0.33, "tail_mean_ccrr": 0.08, "tail_median_ccrr": 0.08, "flagged": False},
        ]
    )
    stratum_concentration_df = pd.DataFrame(
        [
            {"model": "A", "tail_sample_count": 10, "tail_strata_count": 3, "flagged_strata_count": 2, "effective_tail_strata": 2.4},
        ]
    )
    label_enrichment_df = pd.DataFrame(
        [
            {"model": "A", "label": "A", "tail_count": 4, "tail_support_threshold": 3, "tail_frac": 0.40, "dataset_count": 4, "dataset_frac": 0.20, "enrichment_ratio": 2.0, "tail_mean_ccrr": 0.20, "tail_median_ccrr": 0.21, "flagged": True},
            {"model": "A", "label": "B", "tail_count": 3, "tail_support_threshold": 3, "tail_frac": 0.30, "dataset_count": 3, "dataset_frac": 0.15, "enrichment_ratio": 2.0, "tail_mean_ccrr": 0.35, "tail_median_ccrr": 0.36, "flagged": True},
        ]
    )
    center_enrichment_df = pd.DataFrame(
        [
            {"model": "A", "medical_center": "C1", "tail_count": 4, "tail_support_threshold": 3, "tail_frac": 0.40, "dataset_count": 4, "dataset_frac": 0.20, "enrichment_ratio": 2.0, "tail_mean_ccrr": 0.20, "tail_median_ccrr": 0.21, "flagged": True},
            {"model": "A", "medical_center": "C2", "tail_count": 3, "tail_support_threshold": 3, "tail_frac": 0.30, "dataset_count": 3, "dataset_frac": 0.15, "enrichment_ratio": 2.0, "tail_mean_ccrr": 0.35, "tail_median_ccrr": 0.36, "flagged": True},
        ]
    )
    slide_enrichment_df = pd.DataFrame(
        [
            {"model": "A", "slide_id": "slide-hot", "tail_count": 3, "tail_support_threshold": 3, "tail_frac": 0.30, "dataset_count": 3, "dataset_frac": 0.08, "enrichment_ratio": 3.75, "tail_mean_ccrr": 0.10, "tail_median_ccrr": 0.10, "flagged": True},
            {"model": "A", "slide_id": "slide-warm", "tail_count": 3, "tail_support_threshold": 3, "tail_frac": 0.30, "dataset_count": 3, "dataset_frac": 0.09, "enrichment_ratio": 3.33, "tail_mean_ccrr": 0.18, "tail_median_ccrr": 0.18, "flagged": True},
        ]
    )
    flagged_slide_df = pd.DataFrame(
        [
            {"model": "A", "label": "A", "medical_center": "C1", "slide_id": "slide-hot", "slide_tail_count": 2, "slide_stratum_count": 2, "tail_frac_within_stratum": 0.50, "stratum_frac": 0.20, "stratum_enrichment_ratio": 2.5, "flagged": True},
            {"model": "A", "label": "B", "medical_center": "C2", "slide_id": "slide-warm", "slide_tail_count": 2, "slide_stratum_count": 2, "tail_frac_within_stratum": 0.67, "stratum_frac": 0.25, "stratum_enrichment_ratio": 2.67, "flagged": True},
        ]
    )

    findings = ar._build_single_model_tail_characterization(
        df_model=metrics_df,
        rank_df=rank_df,
        action_flags_df=action_flags_df,
        ccrr_stratum_enrichment_df=stratum_enrichment_df,
        ccrr_stratum_concentration_df=stratum_concentration_df,
        ccrr_label_enrichment_df=label_enrichment_df,
        ccrr_center_enrichment_df=center_enrichment_df,
        ccrr_slide_enrichment_df=slide_enrichment_df,
        ccrr_flagged_stratum_slide_concentration_df=flagged_slide_df,
    )

    assert set(findings["question_scope"]) == {
        "reliability",
        "tail_concentration",
        "stratum_enrichment",
        "label_enrichment",
        "center_enrichment",
        "stratum_severity",
        "slide_enrichment",
        "stratum_slide_enrichment",
    }
    assert len(findings.loc[findings["question_scope"] == "stratum_enrichment"]) == 2
    assert len(findings.loc[findings["question_scope"] == "label_enrichment"]) == 2
    assert len(findings.loc[findings["question_scope"] == "center_enrichment"]) == 2
    assert len(findings.loc[findings["question_scope"] == "slide_enrichment"]) == 2
    severity = findings.loc[findings["question_scope"] == "stratum_severity"].sort_values("report_rank_within_model")
    assert list(severity["label"]) == ["C", "A", "B"]
    assert list(severity["medical_center"]) == ["C3", "C1", "C2"]
    assert list(severity["tail_mean_ccrr"]) == [0.08, 0.20, 0.35]
    concentration = findings.loc[findings["question_scope"] == "tail_concentration"].iloc[0]
    assert concentration["effective_tail_strata"] == pytest.approx(2.4)
    assert concentration["flagged_strata_count"] == 2


def test_single_model_tail_characterization_omits_empty_risk_sections(monkeypatch, tmp_path: Path) -> None:
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
        ],
    )
    assert ar.main() == 0

    report = (out_dir / "single_model_tail_characterization.md").read_text(encoding="utf-8")
    assert "### Stratum Enrichment" not in report
    assert "### Biology Enrichment" not in report
    assert "### Center Enrichment" not in report
    assert "### Slide-Level Patterns" not in report
    assert "## Model A" in report
    assert "### Tail Concentration" in report


def test_analyze_results_detailed_writes_ccrr_slide_diagnostics_outputs(monkeypatch, tmp_path: Path) -> None:
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
            "--detailed",
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

    report = (out_dir / "single_model_tail_characterization.md").read_text(encoding="utf-8")
    assert "### Slide-Level Patterns" in report
    assert "slide-hot" in report


def test_ccrr_slide_diagnostics_require_stricter_support_than_strata() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": "A",
               
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
            {"dataset": "toy", "model": "A", "sample_id": "s0", "slide_id": "slide-hot", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.10},
            {"dataset": "toy", "model": "A", "sample_id": "s1", "slide_id": "slide-hot", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.11},
            {"dataset": "toy", "model": "A", "sample_id": "s2", "slide_id": "slide-other-a", "label": "A", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.50},
            {"dataset": "toy", "model": "A", "sample_id": "s3", "slide_id": "slide-other-a", "label": "A", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.51},
            {"dataset": "toy", "model": "A", "sample_id": "s4", "slide_id": "slide-other-b", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.60},
            {"dataset": "toy", "model": "A", "sample_id": "s5", "slide_id": "slide-other-b", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.61},
            {"dataset": "toy", "model": "A", "sample_id": "s6", "slide_id": "slide-other-c", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.62},
            {"dataset": "toy", "model": "A", "sample_id": "s7", "slide_id": "slide-other-c", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.63},
            {"dataset": "toy", "model": "A", "sample_id": "s8", "slide_id": "slide-other-d", "label": "B", "medical_center": "C1", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.64},
            {"dataset": "toy", "model": "A", "sample_id": "s9", "slide_id": "slide-other-d", "label": "B", "medical_center": "C2", "k": 3, "tau": 0.2, "ccrr_alpha": 0.20, "ccrr_search": "start=200;growth=2;alpha=0.2", "excluded_centers": "", "ccrr_m1": 0.65},
        ]
    )

    enrichment_df, summary_df = ar._ccrr_slide_diagnostics(metrics_df, per_sample_df)

    target = enrichment_df.loc[enrichment_df["slide_id"] == "slide-hot"].iloc[0]
    assert target["tail_count"] == 2
    assert target["enrichment_ratio"] > 3.0
    assert not bool(target["flagged"])
    assert summary_df.iloc[0]["flagged_slides_count"] == 0


def test_analyze_results_detailed_adds_sweep_insights_and_flags(monkeypatch, tmp_path: Path) -> None:
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
            {"model": "A", "m": 1, "ccrr": 0.40, "ccrr_q_alpha": 0.30, "ccrr_ltm_alpha": 0.25, "ccrr_retries": 11, "ccrr_k_final": 12800},
            {"model": "A", "m": 10, "ccrr": 0.58, "ccrr_q_alpha": 0.41, "ccrr_ltm_alpha": 0.35, "ccrr_retries": 6, "ccrr_k_final": 7000},
            {"model": "B", "m": 1, "ccrr": 0.70, "ccrr_q_alpha": 0.66, "ccrr_ltm_alpha": 0.62, "ccrr_retries": 5, "ccrr_k_final": 4500},
            {"model": "B", "m": 10, "ccrr": 0.74, "ccrr_q_alpha": 0.69, "ccrr_ltm_alpha": 0.66, "ccrr_retries": 4, "ccrr_k_final": 4300},
            {"model": "C", "m": 1, "ccrr": 0.90, "ccrr_q_alpha": 0.55, "ccrr_ltm_alpha": 0.48, "ccrr_retries": 10, "ccrr_k_final": 9800},
            {"model": "C", "m": 10, "ccrr": 0.93, "ccrr_q_alpha": 0.61, "ccrr_ltm_alpha": 0.57, "ccrr_retries": 6, "ccrr_k_final": 6400},
        ]
    ).to_csv(tmp_path / "ccrr_m_sweep_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": "A", "m": 1, "ccrr": 0.40, "ccrr_q_alpha": 0.30, "ccrr_ltm_alpha": 0.25, "ccrr_retries": 11, "ccrr_k_final": 12800},
            {"model": "A", "m": 10, "ccrr": 0.58, "ccrr_q_alpha": 0.41, "ccrr_ltm_alpha": 0.35, "ccrr_retries": 6, "ccrr_k_final": 7000},
            {"model": "B", "m": 1, "ccrr": 0.70, "ccrr_q_alpha": 0.66, "ccrr_ltm_alpha": 0.62, "ccrr_retries": 5, "ccrr_k_final": 4500},
            {"model": "B", "m": 10, "ccrr": 0.74, "ccrr_q_alpha": 0.69, "ccrr_ltm_alpha": 0.66, "ccrr_retries": 4, "ccrr_k_final": 4300},
            {"model": "C", "m": 1, "ccrr": 0.90, "ccrr_q_alpha": 0.55, "ccrr_ltm_alpha": 0.48, "ccrr_retries": 10, "ccrr_k_final": 9800},
            {"model": "C", "m": 10, "ccrr": 0.93, "ccrr_q_alpha": 0.61, "ccrr_ltm_alpha": 0.57, "ccrr_retries": 6, "ccrr_k_final": 6400},
        ]
    ).to_csv(tmp_path / "ccrr_m_sweep_metrics.csv", index=False)
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
            "--detailed",
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
    assert not (flags["flag"] == "high_undefined_ri").any()
    assert (flags["flag"] == "tail_gap_ltm_high").any()
    assert (flags["flag"] == "k_sweep_sensitivity_high").any()
    assert (flags["flag"] == "ccrr_search_cost_high").any()

    report = (out_dir / "single_model_tail_characterization.md").read_text(encoding="utf-8")
    assert "### Reliability" in report


def test_model_action_flags_ignore_cluster_entanglement_breakdown() -> None:
    df_model = pd.DataFrame(
        [
            {
                "model": "A",
                "ri": 0.80,
                "mari": 0.78,
                "ccrr": 0.70,
                "ri_undefined_frac": 0.20,
                "ri_ss_dominated_undefined_frac": 0.18,
                "ri_oo_dominated_undefined_frac": 0.00,
                "mari_undefined_frac": 0.25,
                "mari_ss_dominated_undefined_frac": 0.22,
                "mari_oo_dominated_undefined_frac": 0.00,
            }
        ]
    )

    flags = ar._model_action_flags(
        df_model=df_model,
        delta_df=pd.DataFrame(),
        k_sensitivity_df=pd.DataFrame(),
        ccrr_m_sensitivity_df=pd.DataFrame(),
    )

    assert len(flags) == 0


def test_single_model_tail_characterization_reports_reliability_summary(monkeypatch, tmp_path: Path) -> None:
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
            {"model": "A", "m": 1, "ccrr": 0.40, "ccrr_q_alpha": 0.30, "ccrr_ltm_alpha": 0.25, "ccrr_retries": 11, "ccrr_k_final": 12800},
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
        ],
    )
    assert ar.main() == 0

    summary_df = pd.read_csv(out_dir / "single_model_tail_characterization.csv")
    reliability = summary_df.loc[summary_df["question_scope"] == "reliability"].copy()
    assert set(reliability["model"]) == {"A"}
    assert "ccrr_search_cost_high" not in reliability.get("flag", pd.Series(dtype=str)).astype(str).tolist()
    assert "coverage_risk_ri" in reliability["flag"].astype(str).tolist()
    assert "coverage_risk_mari" in reliability["flag"].astype(str).tolist()
    report = (out_dir / "single_model_tail_characterization.md").read_text(encoding="utf-8")
    assert "### Reliability" in report
    assert "RI coverage is poor" in report
    assert "MaRI coverage is poor" in report
    assert "CCRR search k_final is high" not in report
    assert "CCRR search is costly" not in report


def test_analyze_results_script_entrypoint_writes_outputs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_metrics().to_csv(metrics_path, index=False)
    out_dir = tmp_path / "analysis"

    result = subprocess.run(
        [
            "/Users/clems/Code/venv/ijepath/bin/python",
            str(ROOT / "scripts" / "analyze_results.py"),
            "--metrics-csv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "single_model_tail_characterization.md").exists()
    assert (out_dir / "single_model_tail_characterization.csv").exists()
    assert (out_dir / "cross_model_findings.csv").exists()
