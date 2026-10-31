"""The typed-neighbour coverage table and its macros read the live Camelyon run.

The table (``tab:typed-neighbour-coverage``) used to be hand-authored from a sealed
20-encoder snapshot. It is now rendered by ``generate_supp_rank_table.py`` from the run the
paper reports -- its metrics plus the two studies written beside it -- so these tests build
a synthetic run under a temporary root and drive the generator's public ``build``/``macros``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts/repro", ROOT / "scripts/bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_supp_rank_table as typed_table  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

CAMELYON = by_benchmark("pathorob-camelyon")

# Registry identities, as the runs store them; the table is a published surface and restyles
# "RudolfV 2-S" to "RudolfV-2-S".
MODELS = ["Alpha", "RudolfV 2-S", "Gamma", "DINOv2-B"]


def _write_run(
    root: Path,
    *,
    k: int = 11,
    contested_k: int | None = None,
    rank_models: list[str] | None = None,
) -> None:
    metrics = pd.DataFrame(
        {
            "model": MODELS,
            "k": [k] * 4,
            "croma": [0.10, 0.30, -0.05, 0.02],
            "support": [0.40, 0.20, 0.10, 0.70],
        }
    )
    metrics_path = root / CAMELYON.metrics_rel
    metrics_path.parent.mkdir(parents=True)
    metrics.to_csv(metrics_path, index=False)

    studies = root / CAMELYON.studies_rel
    studies.mkdir(parents=True)
    ranks = pd.DataFrame(
        {
            "model": rank_models or MODELS,
            "so_med": [100.4, 50.6, 900.0, 30.0],
            "os_med": [400.0, 300.2, 120.0, 40.0],
        }
    )
    ranks.to_csv(studies / "typed_neighbor_rank_summary.csv", index=False)
    (studies / "typed_neighbor_rank_summary.json").write_text(
        json.dumps(
            {
                "ranked_n_models": 3,
                "n_samples": 20400,
                "pooled_first_percentiles_ranked": {"50": 148.6},
            }
        )
    )
    pd.DataFrame(
        {
            "model": MODELS,
            "k": [contested_k or k] * 4,
            "defined_frac": [0.40, 0.20, 0.10, 0.70],
            "pure_so_frac_of_defined": [0.80, 0.70, 0.10, 0.40],
            "pure_os_frac_of_defined": [0.15, 0.20, 0.88, 0.30],
            "contested_frac_of_defined": [0.05, 0.10, 0.02, 0.30],
        }
    ).to_csv(studies / "contested_fraction_summary.csv", index=False)


def _model_rows(tex: str) -> list[str]:
    return re.findall(r"^([^&\n]+?) & [-$0-9.]+ & ", tex, flags=re.MULTILINE)


def test_coverage_table_lists_the_live_roster_by_croma_with_the_control_last(tmp_path):
    _write_run(tmp_path)

    tex = typed_table.build(tmp_path)

    assert _model_rows(tex) == ["RudolfV-2-S", "Alpha", "Gamma", "DINOv2-B"]
    assert r"\label{tab:typed-neighbour-coverage}" in tex
    # SO/OS median ranks, support, then the pure-SO / pure-OS / contested partition (%).
    assert "RudolfV-2-S & 0.30 & 51 & 300 & 20.0 & 70.0 & 20.0 & 10.0 \\\\" in tex


def test_coverage_caption_states_the_runs_operating_k_and_pooled_first_typed_rank(tmp_path):
    _write_run(tmp_path, k=13, contested_k=13)

    tex = typed_table.build(tmp_path)

    assert "$k{=}13$" in tex
    assert "$k{=}11$" not in tex
    assert r"approximately $149$" in tex
    assert "$20{,}400$ tiles" in tex
    assert "three pathology encoders" in tex


def test_typed_neighbour_macros_quote_the_first_typed_rank_and_mixed_share(tmp_path):
    _write_run(tmp_path)

    macros = typed_table.macros(tmp_path)

    assert r"\newcommand{\CamelyonFirstTypedRankMedian}{149}" in macros
    # Median and maximum over the three ranked encoders; the control's 30% is excluded.
    assert r"\newcommand{\CamelyonContestedPctMedian}{5.0\%}" in macros
    assert r"\newcommand{\CamelyonContestedPctMax}{10.0\%}" in macros


def test_a_study_from_a_different_roster_fails_loudly(tmp_path):
    _write_run(tmp_path, rank_models=["Alpha", "Beta", "Gamma", "DINOv2-B"])

    with pytest.raises(SystemExit, match="roster"):
        typed_table.build(tmp_path)


def test_a_contested_study_at_another_k_fails_loudly(tmp_path):
    _write_run(tmp_path, k=11, contested_k=9)

    with pytest.raises(SystemExit, match="k"):
        typed_table.macros(tmp_path)


def test_a_missing_study_fails_loudly(tmp_path):
    _write_run(tmp_path)
    (tmp_path / CAMELYON.studies_rel / "contested_fraction_summary.csv").unlink()

    with pytest.raises(SystemExit, match="contested_fraction_summary"):
        typed_table.macros(tmp_path)
