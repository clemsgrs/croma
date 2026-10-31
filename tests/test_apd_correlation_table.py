"""tab:apd-correlation renders rho, a CI and q per association, from a hand-built CSV frame."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

import generate_apd_floats as gaf  # noqa: E402
from _apd import Apd  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

TILE = ("camelyon", "tcga_4x4", "tolkach")
PLAIN = ("croma", "ri", "mari")
TAILS = ("croma_ltm_alpha", "croma_f0")


def _row(target, metric, scope, **values):
    tile = scope in TILE
    base = dict(
        target=target,
        metric=metric,
        scope=scope,
        n=26 if tile else 5,
        partial_on="croma" if metric.endswith("_given_croma") else "",
        spearman=0.5,
        spearman_p=0.01,
        perm_p=0.02,
        ci_lo=0.1,
        ci_hi=0.8,
        permutation="monte-carlo" if tile else "exact",
        n_permutations=10000 if tile else 120,
        n_boot=2000,
        level=0.95,
        in_bh_family=tile,
        q_bh=0.03 if tile else np.nan,
        two_level_lo=np.nan,
        two_level_hi=np.nan,
    )
    return {**base, **values}


def _frame():
    rows = []
    for scope in (*TILE, "pcabiop"):
        for metric in (*PLAIN, *TAILS):
            rows.append(_row("nipd_id", metric, scope))
        if scope in TILE:
            rows += [_row("nipd_id", f"{m}_given_croma", scope) for m in TAILS]
        rows += [_row("nipd_ood", m, scope) for m in PLAIN]
        for target in ("apd_id", "apd_ood"):
            rows.append(_row(target, "croma", scope))
            # Computed but not rendered, so outside the family.
            rows += [_row(target, m, scope, in_bh_family=False, q_bh=np.nan) for m in PLAIN[1:]]
    frame = pd.DataFrame(rows)
    at = (frame.target == "nipd_id") & (frame.metric == "croma") & (frame.scope == "camelyon")
    frame.loc[at, ["spearman", "ci_lo", "ci_hi", "q_bh", "two_level_lo", "two_level_hi"]] = [
        0.946,
        0.8344,
        0.9774,
        0.0001,
        0.8202,
        0.9659,
    ]
    at = (frame.target == "nipd_id") & (frame.metric == "croma") & (frame.scope == "pcabiop")
    frame.loc[at, ["spearman", "ci_lo", "ci_hi", "perm_p"]] = [0.9, 0.1111, 1.0, 10 / 120]
    at = (frame.target == "nipd_id") & (frame.metric == "croma_f0_given_croma")
    frame.loc[at & (frame.scope == "tolkach"), ["spearman", "ci_lo", "ci_hi", "q_bh"]] = [
        -0.614,
        -0.83,
        -0.25,
        0.0123,
    ]
    return frame


class _Apd(Apd):
    def assert_control_excluded(self) -> None:  # the join is not on disk here
        pass


def _lines(tex: str) -> list[str]:
    return tex.splitlines()


def _line(tex: str, label: str) -> str:
    (line,) = [line for line in _lines(tex) if label in line]
    return line


def test_every_tile_cell_carries_rho_its_interval_and_q():
    tex = gaf.build_table(_Apd(corr=_frame()), design="encoder")
    line = _line(tex, r"\multirow{7}{*}{\code{nIPD}$_\mathrm{ID}$}")
    assert line.split(" & ")[2] == r"$0.95$\,{\scriptsize$[0.83, 0.98]$; $q{<}0.001$}"
    partial = _line(tex, r"$F(0)\,|\,$median")
    assert r"$-0.61$\,{\scriptsize$[-0.83, -0.25]$; $q{=}0.01$}" in partial


def test_pcabiop_is_a_descriptive_column_with_an_exact_p_and_no_q():
    tex = gaf.build_table(_Apd(corr=_frame()), design="encoder")
    assert r" & Metric & Camelyon & TCGA-4x4 & Tolkach-ESCA & PCaBiop \\" in tex
    line = _line(tex, r"\multirow{7}{*}{\code{nIPD}$_\mathrm{ID}$}")
    assert line.split(" & ")[-1] == r"$0.90$\,{\scriptsize$[0.11, 1.00]$; $p{=}0.08$} \\"
    assert _line(tex, r"$\mcode{LTM}_{10}\,|\,$median").endswith(r"& --- \\")


def test_the_tail_rows_are_plain_and_partial_under_nipd_id_only():
    tex = gaf.build_table(_Apd(corr=_frame()), design="encoder")
    for label in (
        r" & $\mcode{LTM}_{10}$ & ",
        r" & $F(0)$ & ",
        r" & $\mcode{LTM}_{10}\,|\,$median & ",
        r" & $F(0)\,|\,$median & ",
    ):
        assert len([line for line in _lines(tex) if label in line]) == 1


def test_the_caption_states_family_replicates_permutations_design_and_panel():
    tex = gaf.build_table(_Apd(corr=_frame()), design="encoder")
    assert "family of the $36$ tile-benchmark associations shown" in tex
    assert "$2000$ replicates" in tex
    assert "$10000$ permutations" in tex
    assert "all $120$ orderings" in tex
    assert "encoder-bootstrap intervals" in tex
    assert "over this panel of encoders" in tex
    assert "Freedman--Lane" in tex


def test_the_switch_renders_the_two_level_interval_for_croma_against_nipd():
    tex = gaf.build_table(_Apd(corr=_frame()), design="two-level")
    line = _line(tex, r"\multirow{7}{*}{\code{nIPD}$_\mathrm{ID}$}")
    assert line.split(" & ")[2] == r"$0.95$\,{\scriptsize$[0.82, 0.97]$; $q{<}0.001$}"
    assert "two-level bootstrap" in tex
    assert gaf.CI_DESIGN in {"encoder", "two-level"}


def test_a_csv_without_the_inference_columns_is_stale():
    with pytest.raises(CaptionClaimError, match="apd_croma_correlation.py"):
        gaf.build_table(_Apd(corr=_frame().drop(columns=["q_bh"])), design="encoder")
