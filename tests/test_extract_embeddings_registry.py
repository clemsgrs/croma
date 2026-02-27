import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_embeddings as ee


def test_registry_includes_conch_and_midnight_models() -> None:
    registry = ee._build_model_registry()

    assert "CONCH" in registry
    assert registry["CONCH"].backend == "conch_v1"
    assert registry["CONCH"].extract == "raw"

    assert "CONCHv1.5" in registry
    assert registry["CONCHv1.5"].backend == "conch_v1_5"
    assert registry["CONCHv1.5"].extract == "raw"
    assert registry["CONCHv1.5"].mixed_precision is True

    assert "Midnight-12k" in registry
    assert registry["Midnight-12k"].backend == "midnight"
    assert registry["Midnight-12k"].model_id == "kaiko-ai/midnight"
    assert registry["Midnight-12k"].extract == "cls_and_patch"


def test_parse_args_accepts_progress_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("auto", "on", "off"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "extract_embeddings.py",
                "--manifest",
                "/tmp/manifest.csv",
                "--output-dir",
                "/tmp/out",
                "--models",
                "UNI",
                "--progress",
                mode,
            ],
        )
        args = ee.parse_args()
        assert args.progress == mode


def test_parse_args_rejects_invalid_progress_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_embeddings.py",
            "--manifest",
            "/tmp/manifest.csv",
            "--output-dir",
            "/tmp/out",
            "--models",
            "UNI",
            "--progress",
            "invalid",
        ],
    )
    with pytest.raises(SystemExit):
        ee.parse_args()
