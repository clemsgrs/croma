
import json
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
        start_k=200,
        k_growth_factor=1.5,
        alpha=0.10,
    )
    assert sig == "start=200;growth=1.5;alpha=0.1"


def test_ccrr_search_signature_alpha_format_is_deterministic() -> None:
    sig = mio.ccrr_search_signature(
        start_k=320,
        k_growth_factor=1.75,
        alpha=0.125,
    )
    assert sig == "start=320;growth=1.75;alpha=0.125"


def test_save_metrics_writes_csv_and_json(tmp_path: Path) -> None:
    rows = [
        {"model": "A", "ri": 0.5, "mari": 0.6},
        {"model": "B", "ri": 0.4, "mari": 0.7},
    ]
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.json"

    mio.save_metrics(rows=rows, csv_path=csv_path, json_path=json_path)

    assert csv_path.exists()
    assert json_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df["model"]) == {"A", "B"}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_save_metrics_accepts_generators(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.json"

    def row_iter():
        yield {"model": "A", "ri": 0.5, "mari": 0.6}
        yield {"model": "B", "ri": 0.4, "mari": 0.7}

    mio.save_metrics(rows=row_iter(), csv_path=csv_path, json_path=json_path)

    df = pd.read_csv(csv_path)
    assert df.to_dict(orient="records") == [
        {"model": "A", "ri": 0.5, "mari": 0.6},
        {"model": "B", "ri": 0.4, "mari": 0.7},
    ]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == [
        {"model": "A", "ri": 0.5, "mari": 0.6},
        {"model": "B", "ri": 0.4, "mari": 0.7},
    ]
