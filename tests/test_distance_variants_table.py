"""``tab:distance-variants`` and its macros, rendered from a hand-built study.

The study's outputs are written here by hand -- a ``measures.csv`` with known values, the
``design.json`` it implies and a ``provenance.json`` over both and over two recorded inputs --
so the loader's refusals and the table's numbers can be checked without ``output/``. Excluded
from the sdist with the rest of the paper tooling (ADR-0012).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts" / "repro", ROOT / "scripts" / "studies", ROOT / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _distance_variants as dvs  # noqa: E402
import distance_variants as study  # noqa: E402
import generate_distance_variants_table as table  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

_VARIANTS = ("cosine", "normalised-euclidean", "euclidean", "whitened")
#: Per benchmark, the median-ranking rho of the three alternatives, and their sign changes.
_RANK_RHO = {
    "pathorob-camelyon": (0.99, 0.95, 0.81),
    "pathorob-tcga-2x2": (0.98, 0.97, 0.62),
    "pathorob-tcga-4x4": (1.00, 0.90, 0.74),
    "pathorob-tolkach-esca": (0.97, 0.88, 0.70),
}
_SIGN_CHANGES = {
    "pathorob-camelyon": (0, 1, 3),
    "pathorob-tcga-2x2": (0, 0, 2),
    "pathorob-tcga-4x4": (0, 0, 1),
    "pathorob-tolkach-esca": (0, 2, 4),
}


def _measures() -> pd.DataFrame:
    rows = []
    for benchmark in dvs.BENCHMARK_ORDER:
        names = list(_VARIANTS) + (
            list(dvs.SENSITIVITY) if benchmark == "pathorob-camelyon" else []
        )
        for name in names:
            row = {"benchmark": benchmark, "variant": name, "n_encoders": 26}
            if name != "cosine":
                k = _VARIANTS.index(name) - 1 if name in _VARIANTS else 2
                rho = _RANK_RHO[benchmark][k] - (0.05 if name in dvs.SENSITIVITY else 0.0)
                row.update(
                    rank_rho_median=rho,
                    rank_rho_median_lo=rho - 0.1,
                    rank_rho_median_hi=min(1.0, rho + 0.01),
                    rank_rho_ltm10=rho - 0.2,
                    rank_rho_ltm10_lo=rho - 0.3,
                    rank_rho_ltm10_hi=rho - 0.1,
                    sign_changes=_SIGN_CHANGES[benchmark][k] + (name in dvs.SENSITIVITY),
                    median_abs_delta_f0=0.0125,
                    median_sign_agreement=0.9734,
                )
            if name.startswith("whitened"):
                row.update(dims_median=40.5, dims_min=12, dims_max=80)
            if benchmark in study.NIPD_DATASETS:
                rho_id = 0.95 - 0.05 * _VARIANTS.index(name) if name in _VARIANTS else 0.5
                row.update(
                    rho_nipd_id=rho_id,
                    rho_nipd_id_lo=rho_id - 0.1,
                    rho_nipd_id_hi=rho_id + 0.02,
                    rho_nipd_id_p=0.0001,
                    rho_nipd_ood=0.70,
                    rho_nipd_ood_lo=0.45,
                    rho_nipd_ood_hi=0.90,
                    rho_nipd_ood_p=0.0123,
                )
            rows.append(row)
    return pd.DataFrame(rows)


#: Camelyon's run, as the stale-cell check reads it: two encoders' pooled cosine scores.
_RUN = "runs/camelyon"
_RUN_METRICS = "model,croma,croma_f0\nE1,0.25,0.125\nE2,-0.0625,0.5\n"


def _write_study(root: Path, repo: Path, measures: pd.DataFrame | None = None) -> None:
    measures = _measures() if measures is None else measures
    root.mkdir(parents=True)
    design = {
        "benchmarks": {
            b: {"variants": [{"name": v} for v in dict.fromkeys(group["variant"])]}
            for b, group in measures.groupby("benchmark", sort=False)
        }
    }
    design["benchmarks"]["pathorob-camelyon"]["results_dir"] = _RUN
    (repo / _RUN).mkdir(parents=True, exist_ok=True)
    (repo / _RUN / "metrics.csv").write_text(_RUN_METRICS)
    (root / study.DESIGN).write_text(json.dumps(design))
    measures.to_csv(root / study.MEASURES, index=False)
    pd.DataFrame(
        {
            "benchmark": ["pathorob-camelyon"] * 2,
            "model": ["E1", "E2"],
            "variant": ["cosine"] * 2,
            "median": [0.25, -0.0625],
            "f0": [0.125, 0.5],
        }
    ).to_csv(root / study.SUMMARY, index=False)
    (repo / "inputs").mkdir(parents=True, exist_ok=True)
    (repo / "inputs" / "metrics.csv").write_text("model,croma\nE1,0.1\n")
    (repo / "inputs" / "huge.npy").write_text("not re-hashed on every build")
    provenance = {
        "design_sha256": study.sha256(root / study.DESIGN),
        "inputs": {
            "inputs/metrics.csv": study.sha256(repo / "inputs" / "metrics.csv"),
            "embeddings:pathorob-camelyon:E1": "0" * 64,
        },
        "outputs": {name: study.sha256(root / name) for name in (study.SUMMARY, study.MEASURES)},
    }
    (root / study.PROVENANCE).write_text(json.dumps(provenance))


@pytest.fixture
def data(tmp_path):
    _write_study(tmp_path / "study", tmp_path)
    return dvs.load(tmp_path / "study", repo=tmp_path)


def test_the_table_reports_every_benchmark_and_alternative_with_cis(data):
    tex = table.build(data)

    assert r"\label{tab:distance-variants}" in tex
    for name in ("Camelyon", "TCGA-2x2", "TCGA-4x4", "Tolkach-ESCA"):
        assert rf"\multirow{{3}}{{*}}{{{name}}}" in tex
    # Tolkach-ESCA, PCA-whitened: rho 0.70 [0.60, 0.71], LTM10 0.50 [0.40, 0.60], 4 changes.
    assert (
        r"& PCA-whitened & $0.70$ {\scriptsize$[0.60, 0.71]$} & $0.50$ {\scriptsize$[0.40, 0.60]$}"
        r" & $4$ & $0.013$ & $97.3\%$ \\"
    ) in tex
    # TCGA-2x2 has no downstream cells, so panel b holds the other three cohorts only.
    assert r"\multirow{4}{*}{TCGA-2x2}" not in tex
    for name in ("Camelyon", "TCGA-4x4", "Tolkach-ESCA"):
        assert rf"\multirow{{4}}{{*}}{{{name}}}" in tex
    assert (
        r"& Cosine (reference) & $0.95$ {\scriptsize$[0.85, 0.97]$} & ${<}0.001$ "
        r"& $0.70$ {\scriptsize$[0.45, 0.90]$} & $0.012$ \\"
    ) in tex
    assert r"Camelyon $40.5$ ($12$--$80$)" in tex
    assert r"Camelyon whitened at $90\%$" in tex and r"at $99\%$" in tex
    assert "$26$ ranked encoders" in tex


def test_the_table_and_macros_rebuild_byte_for_byte(data):
    assert table.build(data) == table.build(data)
    assert table.macro_lines(data) == table.macro_lines(data)


def test_macros_carry_the_minimum_rank_rho_the_sign_changes_and_the_nipd_rho(data):
    lines = table.macro_lines(data)

    assert r"\newcommand{\DistanceMinRankRhoNormalizedEuclidean}{0.97}" in lines
    assert r"\newcommand{\DistanceMinRankRhoRawEuclidean}{0.88}" in lines
    assert r"\newcommand{\DistanceMinRankRhoWhitened}{0.62}" in lines
    assert r"\newcommand{\DistanceMinTailRankRhoWhitened}{0.42}" in lines
    # 0+1+3 + 0+0+2 + 0+0+1 + 0+2+4; Camelyon's 90%/99% sensitivity is not counted.
    assert r"\newcommand{\DistanceSignChanges}{13}" in lines
    assert r"\newcommand{\DistanceNipdIdRhoCosineCamelyon}{0.95}" in lines
    assert r"\newcommand{\DistanceNipdIdRhoWhitenedTolkach}{0.80}" in lines
    assert not any("TcgaTwoByTwo" in line for line in lines)


def test_a_study_that_has_not_been_summarised_fails_the_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="distance_variants.py"):
        dvs.load(tmp_path / "nowhere", repo=tmp_path)


def test_an_output_edited_after_the_study_wrote_it_is_refused(tmp_path):
    _write_study(tmp_path / "study", tmp_path)
    measures = pd.read_csv(tmp_path / "study" / study.MEASURES)
    measures.loc[1, "sign_changes"] = 0
    measures.to_csv(tmp_path / "study" / study.MEASURES, index=False)

    with pytest.raises(CaptionClaimError, match="measures.csv"):
        dvs.load(tmp_path / "study", repo=tmp_path)


def test_an_input_that_changed_after_the_summary_is_refused(tmp_path):
    _write_study(tmp_path / "study", tmp_path)
    (tmp_path / "inputs" / "metrics.csv").write_text("model,croma\nE1,0.2\n")

    with pytest.raises(CaptionClaimError, match="metrics.csv"):
        dvs.load(tmp_path / "study", repo=tmp_path)


def test_a_study_missing_a_variant_is_refused(tmp_path):
    measures = _measures()
    measures = measures[
        ~((measures["benchmark"] == "pathorob-tcga-4x4") & (measures["variant"] == "whitened"))
    ]
    _write_study(tmp_path / "study", tmp_path, measures)

    with pytest.raises(CaptionClaimError, match="pathorob-tcga-4x4"):
        dvs.load(tmp_path / "study", repo=tmp_path)


def test_cells_scored_before_the_run_was_rescored_are_refused(tmp_path):
    _write_study(tmp_path / "study", tmp_path)
    (tmp_path / _RUN / "metrics.csv").write_text(_RUN_METRICS.replace("0.25", "0.2500001"))

    with pytest.raises(CaptionClaimError, match="re-scored"):
        dvs.load(tmp_path / "study", repo=tmp_path)
