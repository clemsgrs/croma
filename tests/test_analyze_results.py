import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_results as ar


def test_aggregate_by_model_separates_evaluation_designs() -> None:
    df = pd.DataFrame(
        {
            "model": ["M1", "M1"],
            "evaluation_design": ["dataset_wide", "paired_2x2"],
            "evaluation_unit": ["sample", "occurrence"],
            "ri": [0.8, 0.4],
            "mari": [0.7, 0.3],
        }
    )

    grouped = ar._aggregate_by_model(df)

    assert set(grouped["model"]) == {
        "M1 [dataset_wide;sample]",
        "M1 [paired_2x2;occurrence]",
    }
    assert grouped.sort_values("ri")["ri"].tolist() == [0.4, 0.8]


def test_k_sweep_sensitivity_separates_evaluation_designs() -> None:
    df = pd.DataFrame(
        {
            "model": ["M1", "M1", "M1", "M1"],
            "evaluation_design": ["dataset_wide", "dataset_wide", "paired_2x2", "paired_2x2"],
            "evaluation_unit": ["sample", "sample", "occurrence", "occurrence"],
            "k": [1, 3, 1, 3],
            "ri": [0.8, 0.7, 0.5, 0.2],
            "mari": [0.75, 0.7, 0.45, 0.1],
        }
    )

    grouped = ar._k_sweep_sensitivity(df)

    assert set(grouped["model"]) == {
        "M1 [dataset_wide;sample]",
        "M1 [paired_2x2;occurrence]",
    }
    paired_row = grouped[grouped["model"] == "M1 [paired_2x2;occurrence]"].iloc[0]
    assert float(paired_row["ri_range"]) == 0.3
