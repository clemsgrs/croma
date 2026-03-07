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
                "ri_undefined_frac": 0.80,
                "mari_undefined_frac": 0.18,
                "ccrr_undefined_frac": 0.00,
                "ccrr_retries": 12,
                "ccrr_k_final": 12000,
            },
        ]
    )


def test_analyze_results_writes_core_outputs(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _toy_metrics().to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {"model": "A", "k": 5, "ri": 0.95, "mari": 0.91},
            {"model": "A", "k": 25, "ri": 0.70, "mari": 0.63},
            {"model": "B", "k": 5, "ri": 0.85, "mari": 0.84},
            {"model": "B", "k": 25, "ri": 0.80, "mari": 0.79},
        ]
    ).to_csv(tmp_path / "k_sweep_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": "A", "m": 1, "ccrr": 0.40, "ccrr_q_alpha": 0.30, "ccrr_ltm_alpha": 0.25, "ccrr_retries": 11, "ccrr_k_final": 11000},
            {"model": "A", "m": 10, "ccrr": 0.58, "ccrr_q_alpha": 0.41, "ccrr_ltm_alpha": 0.35, "ccrr_retries": 6, "ccrr_k_final": 7000},
            {"model": "B", "m": 1, "ccrr": 0.70, "ccrr_q_alpha": 0.66, "ccrr_ltm_alpha": 0.62, "ccrr_retries": 5, "ccrr_k_final": 4500},
            {"model": "B", "m": 10, "ccrr": 0.74, "ccrr_q_alpha": 0.69, "ccrr_ltm_alpha": 0.66, "ccrr_retries": 4, "ccrr_k_final": 4300},
        ]
    ).to_csv(tmp_path / "ccrr_m_sweep_metrics.csv", index=False)
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

    assert ar.main() == 0

    for path in (
        out_dir / "correlation_pearson.csv",
        out_dir / "correlation_spearman.csv",
        out_dir / "model_ranks.csv",
        out_dir / "top_models_by_metric.csv",
        out_dir / "rank_deltas.csv",
        out_dir / "rank_agreement.csv",
        out_dir / "model_action_flags.csv",
        out_dir / "analysis_report.md",
        out_dir / "k_sweep_sensitivity.csv",
        out_dir / "ccrr_m_sweep_sensitivity.csv",
    ):
        assert path.exists(), f"Missing output file: {path}"

    ranks = pd.read_csv(out_dir / "model_ranks.csv")
    assert set(c for c in ranks.columns if c.startswith("rank_")) == {"rank_ri", "rank_mari", "rank_ccrr"}
    assert ranks.sort_values("rank_ri").iloc[0]["model"] == "A"


def test_action_flag_helpers_surface_current_risks() -> None:
    df_model = _toy_metrics()
    rank_df = ar._rank_table(df_model, metrics=["ri", "mari", "ccrr"])
    delta_df = ar._rank_deltas(rank_df)
    k_sensitivity_df = ar._k_sweep_sensitivity(
        pd.DataFrame(
            [
                {"model": "A", "k": 5, "ri": 0.95, "mari": 0.91},
                {"model": "A", "k": 25, "ri": 0.70, "mari": 0.63},
                {"model": "B", "k": 5, "ri": 0.85, "mari": 0.84},
                {"model": "B", "k": 25, "ri": 0.80, "mari": 0.79},
            ]
        )
    )
    ccrr_m_sensitivity_df = ar._ccrr_m_sweep_sensitivity(
        pd.DataFrame(
            [
                {"model": "A", "m": 1, "ccrr": 0.40, "ccrr_q_alpha": 0.30, "ccrr_ltm_alpha": 0.25, "ccrr_retries": 11, "ccrr_k_final": 11000},
                {"model": "A", "m": 10, "ccrr": 0.58, "ccrr_q_alpha": 0.41, "ccrr_ltm_alpha": 0.35, "ccrr_retries": 6, "ccrr_k_final": 7000},
                {"model": "B", "m": 1, "ccrr": 0.70, "ccrr_q_alpha": 0.66, "ccrr_ltm_alpha": 0.62, "ccrr_retries": 5, "ccrr_k_final": 4500},
                {"model": "B", "m": 10, "ccrr": 0.74, "ccrr_q_alpha": 0.69, "ccrr_ltm_alpha": 0.66, "ccrr_retries": 4, "ccrr_k_final": 4300},
            ]
        )
    )

    flags = ar._model_action_flags(
        df_model=df_model,
        delta_df=delta_df,
        k_sensitivity_df=k_sensitivity_df,
        ccrr_m_sensitivity_df=ccrr_m_sensitivity_df,
    )

    assert (flags["flag"] == "coverage_risk_ri").any()
    assert (flags["flag"] == "tail_gap_ltm_high").any()
    assert (flags["flag"] == "k_sweep_sensitivity_high").any()
    assert (flags["flag"] == "ccrr_search_cost_high").any()
