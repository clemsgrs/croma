import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import prostate_granularity_analysis as pga  # noqa: E402


def test_per_model_table_preserves_shared_support(tmp_path: Path, monkeypatch) -> None:
    metrics = pd.DataFrame(
        {
            "model": ["M"],
            "croma": [0.2],
            "ri": [0.3],
            "mari": [0.4],
            "support": [0.23],
            "bio_knn_bacc": [0.8],
            "confounder_knn_bacc": [0.7],
        }
    )
    settings = {"natbin": "natural", "gradebal": "balanced", "fourclass": "four"}
    for relative in settings.values():
        results = tmp_path / relative / "results"
        results.mkdir(parents=True)
        metrics.to_csv(results / "metrics.csv", index=False)
    monkeypatch.setattr(pga, "REPO", tmp_path)
    monkeypatch.setattr(pga, "SETTINGS", settings)

    table = pga.per_model_table()

    assert table.loc["M", ["support_natbin", "support_gradebal", "support_fourclass"]].tolist() == [
        0.23,
        0.23,
        0.23,
    ]
