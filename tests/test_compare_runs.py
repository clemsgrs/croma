import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare_runs as cr  # noqa: E402


def test_comparison_cli_reports_positive_shared_support(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    run_a = pd.DataFrame(
        {
            "dataset": ["cohort"],
            "model": ["M"],
            "confounder_column": ["center"],
            "evaluation_design": ["all"],
            "ri": [0.2],
            "mari": [0.3],
            "support": [0.2],
        }
    )
    run_b = run_a.assign(ri=0.4, mari=0.5, support=0.8)

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    output_dir = tmp_path / "comparison"
    for directory, frame in ((old_dir, run_a), (new_dir, run_b)):
        (directory / "results").mkdir(parents=True)
        frame.to_csv(directory / "results" / "metrics.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runs.py",
            "--run-a",
            str(old_dir),
            "--run-b",
            str(new_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert cr.main() == 0

    output = capsys.readouterr().out
    assert "support% (old)" in output
    assert "support% (new)" in output
    assert "undef%" not in output
    comparison = pd.read_csv(output_dir / "comparison.csv")
    assert comparison.to_dict(orient="records") == [
        {
            "model": "M",
            "ri (old)": 0.2,
            "ri (new)": 0.4,
            "Δri": 0.2,
            "mari (old)": 0.3,
            "mari (new)": 0.5,
            "Δmari": 0.2,
            "support% (old)": 20.0,
            "support% (new)": 80.0,
        }
    ]
