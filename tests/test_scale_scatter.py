"""Tests for the scale-vs-robustness scatter (issue #63).

The data-assembly function ``build_scale_frame`` is exercised on synthetic
metadata + synthetic per-dataset CRoMa (no filesystem). Rendering is checked only
for "produces a file without error".
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "scripts" / "repro" / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import scale_scatter as ss  # noqa: E402


DATASETS = ("Camelyon", "TCGA-4x4", "Tolkach")


def _synthetic_metadata() -> pd.DataFrame:
    """Three tile models (one VLFM, one with undisclosed params) + one slide model.

    Cells use the same LaTeX-decorated form as the real ``model_metadata.csv`` so
    the parser is exercised, not bypassed.
    """
    rows = [
        dict(model="Alpha", panel="tile", panel_order=1,
             params="$632$M", n_wsis="$3.1$M", regime="vision-only"),
        dict(model="Beta", panel="tile", panel_order=2,
             params="$86$M", n_wsis="$21.4$k", regime="vision--language"),
        dict(model="Gamma", panel="tile", panel_order=3,
             params="n/d", n_wsis="$100$k", regime="VLFM"),
        dict(model="SlideOnly", panel="slide", panel_order=1,
             params="$100$M", n_wsis="$50$k", regime="vision-only"),
    ]
    return pd.DataFrame(rows)


def _synthetic_croma() -> dict[str, dict[str, float]]:
    return {
        "Camelyon": {"Alpha": 0.10, "Beta": 0.40, "Gamma": -0.20, "SlideOnly": 0.9},
        "TCGA-4x4": {"Alpha": 0.20, "Beta": 0.50, "Gamma": -0.10, "SlideOnly": 0.9},
        "Tolkach": {"Alpha": 0.30, "Beta": 0.60, "Gamma": 0.00, "SlideOnly": 0.9},
    }


def test_one_row_per_tile_model_excludes_slide() -> None:
    frame = ss.build_scale_frame(_synthetic_metadata(), _synthetic_croma())
    assert list(frame["model"]) == ["Alpha", "Beta", "Gamma"]
    assert "SlideOnly" not in set(frame["model"])
    assert len(frame) == 3


def test_cross_dataset_mean_croma() -> None:
    frame = ss.build_scale_frame(_synthetic_metadata(), _synthetic_croma())
    by_model = frame.set_index("model")["croma_mean"]
    assert by_model["Alpha"] == pytest.approx((0.10 + 0.20 + 0.30) / 3)
    assert by_model["Beta"] == pytest.approx((0.40 + 0.50 + 0.60) / 3)
    assert by_model["Gamma"] == pytest.approx((-0.20 - 0.10 + 0.00) / 3)
    assert set(frame["n_datasets"]) == {3}


def test_regime_label_mapping() -> None:
    frame = ss.build_scale_frame(_synthetic_metadata(), _synthetic_croma())
    regime = frame.set_index("model")["regime"]
    assert regime["Alpha"] == "vision-only"
    assert regime["Beta"] == "VLFM"  # from "vision--language"
    assert regime["Gamma"] == "VLFM"  # already "VLFM"


def test_params_and_wsis_parsed() -> None:
    frame = ss.build_scale_frame(_synthetic_metadata(), _synthetic_croma())
    by_model = frame.set_index("model")
    assert by_model.loc["Alpha", "params"] == pytest.approx(632e6)
    assert by_model.loc["Alpha", "n_wsis"] == pytest.approx(3.1e6)
    assert by_model.loc["Beta", "params"] == pytest.approx(86e6)
    assert by_model.loc["Beta", "n_wsis"] == pytest.approx(21400.0)
    assert by_model.loc["Gamma", "n_wsis"] == pytest.approx(100_000.0)
    # Undisclosed params survive as NaN (drawn hollow by the renderer).
    assert math.isnan(by_model.loc["Gamma", "params"])


def test_model_missing_in_a_dataset_is_dropped() -> None:
    croma = _synthetic_croma()
    del croma["Tolkach"]["Beta"]  # Beta now absent from one benchmark
    frame = ss.build_scale_frame(_synthetic_metadata(), croma)
    assert "Beta" not in set(frame["model"])
    assert list(frame["model"]) == ["Alpha", "Gamma"]


def test_parse_scale_latex_forms() -> None:
    assert ss._parse_scale("${\\sim}1.1$B") == pytest.approx(1.1e9)
    assert ss._parse_scale("${>}1$M") == pytest.approx(1e6)
    assert ss._parse_scale("$6{,}093$") == pytest.approx(6093.0)
    assert ss._parse_scale("$500$K") == pytest.approx(500e3)
    assert math.isnan(ss._parse_scale("n/d"))


def test_render_produces_a_file(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")  # noqa: F841
    frame = ss.build_scale_frame(_synthetic_metadata(), _synthetic_croma())
    out = ss.render_scale_scatter(frame, tmp_path / "scale_scatter.pdf")
    assert out.exists()
    assert (out.parent / "png" / "scale_scatter.png").exists()
