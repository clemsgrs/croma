import sys
from pathlib import Path

import pandas as pd

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
