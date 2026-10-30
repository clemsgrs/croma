"""The threshold-free nIPD study's paper surfaces: its basis, ``tab:nipd-auroc``, macros.

Everything renders from synthetic reductions or an empty ``tmp_path`` root; no ``paper/``
or ``output/`` tree is needed. The study itself is tested in ``test_auc_study.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "repro", ROOT / "scripts" / "bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _nipd_auroc as basis  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

TILE_K = {"camelyon": 11, "tcga_4x4": 81, "tolkach": 71}
SLIDE_K = {"MOOZY": 3, "PRISM": 7, "PRISM2": 9, "Prov-GigaPath": 13, "TITAN": 9}
#: Small resampling counts keep the tests fast; the caption must report whichever were used.
FAST = dict(n_boot=200, n_permutations=500)


def _reduction(n_ranked: int = 26, *, controls: dict[str, int] | None = None) -> pd.DataFrame:
    """A reduction shaped like the study's: ``n_ranked`` encoders plus the control on each
    tile cohort, the five slide encoders on PCaBiop."""
    rng = np.random.default_rng(7)
    controls = controls or {}
    rows = []
    panels = {d: [f"Enc{i:02d}" for i in range(n_ranked)] for d in TILE_K}
    panels["pcabiop"] = list(SLIDE_K)
    for dataset, models in panels.items():
        n_controls = controls.get(dataset, 0 if dataset == "pcabiop" else 1)
        models = models + ["DINOv2-B"] * n_controls
        for regime in ("id", "ood"):
            base = rng.normal(size=len(models))
            for i, model in enumerate(models):
                rows.append(
                    dict(
                        dataset=dataset,
                        model=model,
                        regime=regime,
                        control=model == "DINOv2-B",
                        croma=base[i] + rng.normal(scale=0.4),
                        ri=base[i] + rng.normal(scale=0.6),
                        mari=base[i] + rng.normal(scale=0.5),
                        k=SLIDE_K[model] if dataset == "pcabiop" else TILE_K[dataset],
                        nipd_auroc=base[i] + rng.normal(scale=0.3),
                        nipd_balacc=base[i] + rng.normal(scale=0.5),
                    )
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def study():
    return basis.from_reduction(_reduction(), **FAST)


class TestBasis:
    def test_a_study_whose_cells_have_not_been_run_fails_loudly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no APD cells"):
            basis.load(tmp_path)

    def test_the_panel_is_every_ranked_encoder_and_never_the_control(self, study):
        assert study.n_ranked == 26
        assert study.n("pcabiop") == 5
        assert study.family_size == 36

    def test_a_tile_cohort_without_its_control_cannot_claim_the_exclusion(self):
        with pytest.raises(CaptionClaimError, match="tolkach"):
            basis.from_reduction(_reduction(controls={"tolkach": 0}), **FAST)

    def test_tile_cohorts_over_different_panels_cannot_share_one_count(self):
        reduction = _reduction()
        reduction = reduction[
            ~((reduction["dataset"] == "tcga_4x4") & (reduction["model"] == "Enc00"))
        ]

        with pytest.raises(CaptionClaimError, match="tcga_4x4"):
            basis.from_reduction(reduction, **FAST)

    def test_the_operating_point_is_the_runs(self, study):
        assert [study.operating_k(d) for d in TILE_K] == [11, 81, 71]
        # Each slide encoder sits at its own k*: no single k to quote.
        assert study.operating_k("pcabiop") is None


def _planted(study, cells: dict[tuple[str, str, str, str], dict]):
    """``study`` with chosen correlations replaced by known values."""
    corr = study.corr.copy()
    for (dataset, regime, x, y), values in cells.items():
        mask = (
            (corr["dataset"] == dataset)
            & (corr["regime"] == regime)
            & (corr["x"] == x)
            & (corr["y"] == y)
        )
        assert mask.sum() == 1
        for column, value in values.items():
            corr.loc[mask, column] = value
    return basis.NipdAuroc(reduction=study.reduction, corr=corr)


def _row(tex: str, lead: str) -> list[str]:
    line = next(line for line in tex.splitlines() if line.startswith(lead))
    return line.removesuffix(r" \\").split(" & ")


def _stack(*lines: str) -> str:
    return r"\begin{tabular}[c]{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"


class TestTable:
    def _tex(self, study) -> str:
        import generate_nipd_auroc_table as table

        return table.build(study)

    def test_columns_are_cohort_by_regime_and_rows_metric_by_score(self, study):
        tex = self._tex(study)

        header = next(line for line in tex.splitlines() if r"\multicolumn{2}{c}" in line)
        labels = [part.split("}{")[-1].rstrip("} \\") for part in header.split(" & ")[2:]]
        assert labels == ["Camelyon", "TCGA-4x4", "Tolkach-ESCA", "PCaBiop"]
        body = [line for line in tex.splitlines() if r"\begin{tabular}[c]" in line]
        # Three metrics x two scores, eight cohort-regime cells each...
        assert [line.split(" & ")[:2] for line in body[:6]] == [
            [r"\textbf{\code{CRoMa}}", "AUROC"],
            ["", "Bal.\\ acc."],
            [r"\code{RI}", "AUROC"],
            ["", "Bal.\\ acc."],
            [r"\code{MaRI}", "AUROC"],
            ["", "Bal.\\ acc."],
        ]
        assert {line.count(" & ") for line in body[:6]} == {9}
        # ...then the agreement row at the bottom, its label spanning both label columns.
        assert len(body) == 7
        assert body[6].startswith(r"\multicolumn{2}{l}{Agreement")
        assert r"\label{tab:nipd-auroc}" in tex

    def test_a_tile_association_carries_rho_its_interval_and_q(self, study):
        planted = _planted(
            study,
            {
                ("camelyon", "id", "croma", "nipd_auroc"): dict(
                    rho=0.950376, ci_lo=0.803637, ci_hi=0.989386, q_value=0.000144
                ),
                ("tolkach", "ood", "croma", "nipd_auroc"): dict(
                    rho=0.249624, ci_lo=-0.229854, ci_hi=0.638064, q_value=0.299405
                ),
            },
        )

        cells = _row(self._tex(planted), r"\textbf{\code{CRoMa}} & AUROC")

        assert cells[2] == _stack(
            "$0.95$", r"{\scriptsize$[0.80, 0.99]$}", r"{\scriptsize$q{<}0.001$}"
        )
        assert cells[7] == _stack(
            "$0.25$", r"{\scriptsize$[-0.23, 0.64]$}", r"{\scriptsize$q{=}0.299$}"
        )

    def test_pcabiop_is_descriptive_rho_with_its_exact_p_and_no_q(self, study):
        planted = _planted(
            study,
            {
                ("pcabiop", "id", "croma", "nipd_auroc"): dict(rho=1.0, p_value=2 / 120),
                ("pcabiop", "ood", "mari", "nipd_balacc"): dict(rho=0.7, p_value=28 / 120),
            },
        )
        tex = self._tex(planted)

        assert _row(tex, r"\textbf{\code{CRoMa}} & AUROC")[8] == _stack(
            "$1.00$", r"{\scriptsize$p{=}0.017$}"
        )
        mari_balacc = [line for line in tex.splitlines() if line.startswith(" & Bal.\\ acc.")][-1]
        assert mari_balacc.removesuffix(r" \\").split(" & ")[9] == _stack(
            "$0.70$", r"{\scriptsize$p{=}0.233$}"
        )

    def test_the_agreement_row_has_rho_and_interval_but_is_outside_the_family(self, study):
        planted = _planted(
            study,
            {
                ("tcga_4x4", "ood", "nipd_auroc", "nipd_balacc"): dict(
                    rho=0.950593, ci_lo=0.851070, ci_hi=0.989060
                ),
            },
        )

        cells = _row(self._tex(planted), r"\multicolumn{2}{l}{Agreement")

        assert len(cells) == 9  # the label spans two columns
        assert cells[4] == _stack("$0.95$", r"{\scriptsize$[0.85, 0.99]$}")
        assert "q{" not in " ".join(cells)

    def test_the_caption_quotes_the_counts_the_study_used(self):
        import generate_nipd_auroc_table as table

        caption = table.build(basis.from_reduction(_reduction(20), **FAST)).split(r"\caption{")[1]

        assert "$20$ ranked encoders" in caption
        assert "$200$ encoder resamples" in caption
        assert "$500$ permutations" in caption
        assert "$36$ associations" in caption
        assert "all $120$ orderings" in caption
        assert "$95\\%$" in caption
        assert "$k{=}11$, $81$ and $71$ on Camelyon, TCGA-4x4 and Tolkach-ESCA" in caption
        assert "macro one-vs-rest" in caption and "chance $0.5$" in caption

    def test_the_table_rebuilds_byte_for_byte(self):
        import generate_nipd_auroc_table as table

        first = table.build(basis.from_reduction(_reduction(), **FAST))
        second = table.build(basis.from_reduction(_reduction(), **FAST))

        assert first == second


class TestMacros:
    def test_agreement_per_cohort_and_regime(self, study):
        import generate_paper_values as gpv

        planted = _planted(
            study,
            {
                ("camelyon", "id", "nipd_auroc", "nipd_balacc"): dict(rho=0.971429),
                ("pcabiop", "ood", "nipd_auroc", "nipd_balacc"): dict(rho=0.1),
                ("tolkach", "ood", "nipd_auroc", "nipd_balacc"): dict(rho=-0.004),
            },
        )

        lines = gpv._nipd_auroc_macros(planted)

        assert r"\newcommand{\NipdAurocAgreementIdCamelyon}{0.97}" in lines
        assert r"\newcommand{\NipdAurocAgreementOodPcabiop}{0.10}" in lines
        assert r"\newcommand{\NipdAurocAgreementOodTolkach}{0.00}" in lines
        assert sum(r"\NipdAurocAgreement" in line for line in lines) == 8

    def test_in_domain_croma_rho_with_its_interval_and_q_or_exact_p(self, study):
        import generate_paper_values as gpv

        planted = _planted(
            study,
            {
                ("tcga_4x4", "id", "croma", "nipd_auroc"): dict(
                    rho=0.873518, ci_lo=0.719937, ci_hi=0.941246, q_value=0.000144
                ),
                ("pcabiop", "id", "croma", "nipd_auroc"): dict(rho=1.0, p_value=2 / 120),
            },
        )

        lines = gpv._nipd_auroc_macros(planted)

        assert r"\newcommand{\NipdAurocIdCromaTcga}{0.87}" in lines
        assert r"\newcommand{\NipdAurocIdCromaTcgaCi}{$[0.72, 0.94]$}" in lines
        assert r"\newcommand{\NipdAurocIdCromaTcgaQ}{$q{<}0.001$}" in lines
        # PCaBiop is descriptive, as in the table: rho and its exact p, no interval or q.
        assert r"\newcommand{\NipdAurocIdCromaPcabiop}{1.00}" in lines
        assert r"\newcommand{\NipdAurocIdCromaPcabiopP}{$p{=}0.017$}" in lines
        assert not [line for line in lines if r"\NipdAurocIdCromaPcabiopCi" in line]
        assert not [line for line in lines if r"\NipdAurocIdCromaPcabiopQ" in line]

    def test_the_ranked_panel_size(self, study):
        import generate_paper_values as gpv

        assert r"\newcommand{\NipdAurocNModels}{26}" in gpv._nipd_auroc_macros(study)

    def test_the_values_file_fails_without_the_study(self, tmp_path):
        # Never warn-and-skip: a missing macro would compile to an undefined control
        # sequence at best and silently drop a sentence's number at worst.
        import generate_paper_values as gpv

        with pytest.raises(FileNotFoundError):
            gpv.build([], tmp_path, "auto")


def test_the_paper_build_never_skips_the_table():
    # build_paper skips a study generator whose required input is absent and reports the
    # build as OK; with no required input the generator always runs, so a missing or stale
    # cell fails the build through _nipd_auroc.load.
    import build_paper

    entries = [s for s in build_paper.STUDIES if s[0] == "generate_nipd_auroc_table.py"]

    assert len(entries) == 1
    assert entries[0][1] is None
    assert "sweep.py" in entries[0][2]
