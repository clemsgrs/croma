r"""Render the sample-shortcut floats and macros from ``output/studies/sample-shortcut/``.

Writes ``paper/sections/supp/sample_shortcut.tex``: one table per tile cohort
(``tab:sample-shortcut`` for Camelyon, ``tab:sample-shortcut-tcga4x4``), a short
descriptive table for the PCaBiop slides (``tab:sample-shortcut-pcabiop``), and the
``fig:margin-montage`` float whose PDF ``scripts/repro/figures/margin_montage_figure.py``
renders into ``output/studies/sample-shortcut/plots/pdf/``. :func:`macros` gives the
numbers the text quotes; ``generate_paper_values.py`` calls it unconditionally.

Everything is read through ``_sample_shortcut.load``, which refuses a study whose inputs
changed since it ran and fails on a study that never ran -- nothing here warns and skips.

Usage:
  python scripts/repro/generate_sample_shortcut_floats.py
  python scripts/repro/generate_sample_shortcut_floats.py --out /tmp/sample_shortcut.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import _sample_shortcut as basis
from _paper_tables import (  # noqa: F401  (CaptionClaimError re-exported)
    CROMA_HEADLINE_M,
    CaptionClaimError,
    bare_num,
    num_math,
    scriptsize_ci,
)
from plotting.style import CONTROL_MODEL

REPO = basis.REPO
OUT = REPO / "paper/sections/supp/sample_shortcut.tex"
LABELS = {
    "camelyon": "tab:sample-shortcut",
    "tcga_4x4": "tab:sample-shortcut-tcga4x4",
    "pcabiop": "tab:sample-shortcut-pcabiop",
}
FIGURE_LABEL = "fig:margin-montage"
FIGURE_PDF = "margin_montage.pdf"
#: Macro family per cohort: the run's own prefix for the tile cohorts, the APD studies'
#: ``Pcabiop`` scope for the slides (the run's ``Panda`` prefix names the benchmark).
MACRO_PREFIX = {"camelyon": "Camelyon", "tcga_4x4": "TcgaFourByFour", "pcabiop": "Pcabiop"}
DASH = "---"


def _ci_cell(row: pd.Series, key: str, *, signed: bool = False) -> str:
    if not bool(row["evaluable"]):
        return DASH
    point = num_math(row[key]) if signed else bare_num(row[key])
    return rf"{point}\,{scriptsize_ci(row[f'{key}_lo'], row[f'{key}_hi'])}"


def _name(row: pd.Series) -> str:
    name = str(row["model"])
    return rf"{name}$^{{\dagger}}$" if row["registry_model"] == CONTROL_MODEL else name


def _row(row: pd.Series, *, colour: bool) -> str:
    cells = [
        _name(row),
        str(int(row["n_flipped"])),
        _ci_cell(row, "auroc_croma"),
        _ci_cell(row, "auroc_bio"),
        _ci_cell(row, "auroc_purity"),
        _ci_cell(row, "diff_croma_bio", signed=True),
        _ci_cell(row, "diff_croma_purity", signed=True),
        num_math(row["rho_drop"]),
        num_math(row["rho_drop_given_margin0"]),
    ]
    if colour:
        cells.append(num_math(row["rho_croma_colour"]))
    return " & ".join(cells) + r" \\"


def _n(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _short(result: basis.CohortResult) -> str:
    return result.run.short_name or result.cohort


def _k_clause(result: basis.CohortResult) -> str:
    ks = sorted({int(k) for k in pd.concat([result.ranked["k"], result.control["k"]])})
    if len(ks) == 1:
        return rf"the $k{{=}}{ks[0]}$"
    return r"each encoder's operating-$k$"


def _unit(result: basis.CohortResult) -> tuple[str, str]:
    return ("tile", "tiles") if result.colour else ("slide", "slides")


def _population(result: basis.CohortResult) -> str:
    one, many = _unit(result)
    info = result.info
    never = int(info["n_never_held_out"])
    never_clause = (
        f"; the remaining ${_n(never)}$ were never held out and have no target"
        if never
        else ""
    )
    return (
        rf"Of the ${_n(info['n_tiles'])}$ {many} the run scores, ${_n(info['n_held_out'])}$ are "
        rf"held out by at least one probe replicate{never_clause}."
    )


def _definitions(result: basis.CohortResult) -> str:
    one, many = _unit(result)
    m = int(CROMA_HEADLINE_M)
    return (
        rf"A held-out {one} is \emph{{flipped}} when the biology probe trained on the balanced "
        rf"split (Cram\'er's $V{{=}}0$) never misclassifies it and the probe trained on the fully "
        rf"confounded split ($V{{=}}1$) misclassifies it in at least half of the replicates that "
        rf"hold it out. {_population(result)} AUROCs are for flipped {many} among those the "
        rf"balanced probe never misclassifies, with pairs formed only within class $\times$ "
        rf"{'centre' if result.colour else 'provider'} cells, for three scores: low "
        rf"\code{{CRoMa}} ($m{{=}}{m}$); a low biology-only margin, the same margin with "
        rf"neighbours typed by class alone from any {'centre' if result.colour else 'provider'}; "
        rf"and high {'centre' if result.colour else 'provider'} purity, the own-"
        rf"{'centre' if result.colour else 'provider'} share of {_k_clause(result)} nearest "
        rf"neighbours. $\Delta$ columns are \code{{CRoMa}} minus each baseline. Brackets are 95\% "
        rf"stratified slide-cluster bootstrap intervals ({int(result.info['n_boot'])} replicates, "
        r"slides resampled within cells, one draw shared by all scores, so differences are "
        rf"paired). AUROCs are shown as {DASH} for encoders with fewer than "
        rf"{int(result.info['min_flipped'])} flipped {many}. $\rho_{{\text{{drop}}}}$ and "
        rf"$\rho_{{\text{{drop}}\mid V=0}}$ are the within-cell Spearman correlations of "
        rf"\code{{CRoMa}} with the drop in true-class probe margin from $V{{=}}0$ to $V{{=}}1$, "
        rf"plain and partial given the $V{{=}}0$ margin, over every held-out {one}."
    )


def _ordering(result: basis.CohortResult) -> str:
    control = (
        r"; the natural-image control ($\dagger$) is shown separately and enters no summary"
        if not result.control.empty
        else ""
    )
    return rf"Encoders are ordered by the run's median \code{{CRoMa}}{control}."


def _caption(result: basis.CohortResult) -> str:
    one, many = _unit(result)
    if result.colour:
        tail = (
            r" $\rho_{\text{colour}}$ is the within-cell Spearman correlation of \code{CRoMa} "
            r"with colour centre-confidence, the out-of-fold probability a colour-only "
            r"logistic classifier (optical density and CIELAB means and standard deviations "
            r"over tissue pixels; slide-grouped five-fold cross-validation) gives the tile's "
            r"own centre."
        )
        title = rf"Per-sample \code{{CRoMa}} and shortcut-induced errors on {_short(result)}."
    else:
        tail = r" Slides are the units, so the slide-cluster bootstrap resamples single slides."
        title = rf"Per-slide \code{{CRoMa}} and shortcut-induced errors on PCaBiop."
    return rf"\textbf{{{title}}} {_definitions(result)}{tail} {_ordering(result)}"


def _footer(result: basis.CohortResult, summary: dict, columns: int) -> list[str]:
    centres = result.samples()["cell"].str.split(" | ", regex=False).str[1].nunique()
    text = (
        rf"Colour-only centre classifier: balanced accuracy "
        rf"{bare_num(summary['colour_balanced_accuracy'])} (chance {bare_num(1 / centres)}). "
        rf"Median over evaluable encoders: colour-confidence AUROC "
        rf"{bare_num(summary['colour_auroc_median'])}; \code{{CRoMa}} AUROC within cell "
        rf"$\times$ colour-confidence quintile {bare_num(summary['croma_given_colour_median'])}, "
        rf"interval above 0.5 for {summary['n_given_colour_above_half']} of "
        rf"{summary['n_evaluable']}."
    )
    return [rf"\multicolumn{{{columns}}}{{p{{0.97\linewidth}}}}{{\scriptsize {text}}} \\"]


def _table(result: basis.CohortResult) -> list[str]:
    colour = result.colour
    summary = basis.summary(result)
    columns = 10 if colour else 9
    one, many = _unit(result)
    last = r"$\rho_{\text{drop}}$ & $\rho_{\text{drop}\mid V=0}$"
    if colour:
        last += r" & $\rho_{\text{colour}}$"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        # Shrink to the text width only when the natural width exceeds it; never enlarge.
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        rf"\begin{{tabular}}{{l{'r' * (columns - 1)}}}",
        r"\hline",
        rf"& & \multicolumn{{3}}{{c}}{{within-cell AUROC for flipped {many}}} "
        r"& \multicolumn{2}{c}{paired difference} & \\",
        r"\cline{3-5}\cline{6-7}",
        rf"Model & flipped & \code{{CRoMa}} & biology-only & purity & $\Delta$ biology-only "
        rf"& $\Delta$ purity & {last} \\",
        r"\hline",
    ]
    lines += [_row(row, colour=colour) for _, row in result.ranked.iterrows()]
    if not result.control.empty:
        lines += [r"\hline", *[_row(row, colour=colour) for _, row in result.control.iterrows()]]
    lines.append(r"\hline")
    if colour:
        lines += [*_footer(result, summary, columns), r"\hline"]
    lines += [
        r"\end{tabular}%",
        r"}",
        rf"\caption{{{_caption(result)}}}",
        rf"\label{{{LABELS[result.cohort]}}}",
        r"\end{table}",
    ]
    return lines


def _cell_text(cell: str) -> str:
    label, centre = cell.split(" | ", 1)
    return f"{label.replace('_', ' ')}, {centre}"


def _figure(results: dict[str, basis.CohortResult]) -> list[str]:
    choices = [basis.montage_choice(results[c]) for c in basis.TILE_COHORTS]
    picked = "; ".join(
        rf"{_short(results[c.cohort])}: {c.model} (AUROC {bare_num(c.auroc)}), "
        rf"{_cell_text(c.cell)}, ${_n(c.n_flipped_in_cell)}$ of ${_n(c.n_in_cell)}$ tiles flipped"
        for c in choices
    )
    n = basis.MONTAGE_TILES
    caption = (
        r"\textbf{The lowest- and highest-\code{CRoMa} tiles of one class $\times$ centre cell.} "
        r"For each tile cohort the encoder is the evaluable pathology encoder at the lower "
        r"median of the panel's within-cell \code{CRoMa} AUROC for flipped tiles, and the cell "
        rf"is its class $\times$ centre cell holding the most flipped tiles ({picked}). Top "
        rf"row: the {n} lowest-\code{{CRoMa}} tiles; bottom row: the {n} highest, among the "
        r"cell's tiles the balanced probe never misclassifies. Each tile is annotated with its "
        r"\code{CRoMa} ($m{=}5$), whether the fully confounded probe flipped it, and its colour "
        r"centre-confidence (the colour-only classifier's out-of-fold probability of the tile's "
        rf"own centre). Definitions as in Table~\ref{{{LABELS['camelyon']}}}."
    )
    return [
        r"\begin{figure}[!htbp]",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{{FIGURE_PDF}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{FIGURE_LABEL}}}",
        r"\end{figure}",
    ]


def load_all(root: Path = REPO) -> dict[str, basis.CohortResult]:
    return {cohort: basis.load(cohort, root) for cohort in basis.COHORTS}


def build(root: Path = REPO) -> str:
    results = load_all(root)
    lines = [
        "% AUTO-GENERATED by scripts/repro/generate_sample_shortcut_floats.py -- do not edit.",
        f"% PDF rendered under {basis.STUDY_REL}/plots/pdf/{FIGURE_PDF}.",
    ]
    for cohort in basis.COHORTS:
        lines += _table(results[cohort])
    lines += _figure(results)
    return "\n".join(lines) + "\n"


def macros(root: Path = REPO) -> list[str]:
    """The numbers the text quotes, per cohort; a missing or stale study raises."""
    out = ["% Per-sample margins vs shortcut-induced errors (tab:sample-shortcut*)"]
    for cohort, result in load_all(root).items():
        s = basis.summary(result)
        p = f"{MACRO_PREFIX[cohort]}SampleShortcut"
        specs = [
            ("NRanked", str(s["n_ranked"])),
            ("NEvaluable", str(s["n_evaluable"])),
            ("HeldOut", _n(s["n_held_out"])),
            ("NeverHeldOut", _n(s["n_never_held_out"])),
            ("FlippedMedian", _n(round(s["flipped_median"]))),
            ("CromaAurocMedian", bare_num(s["croma_auroc_median"])),
            ("BioAurocMedian", bare_num(s["bio_auroc_median"])),
            ("PurityAurocMedian", bare_num(s["purity_auroc_median"])),
            ("NCiAboveHalf", str(s["n_ci_above_half"])),
            ("NCiBelowHalf", str(s["n_ci_below_half"])),
            ("NAboveBio", str(s["n_above_bio"])),
            ("NBelowBio", str(s["n_below_bio"])),
            ("NAbovePurity", str(s["n_above_purity"])),
            ("NBelowPurity", str(s["n_below_purity"])),
        ]
        if result.colour:
            specs += [
                ("ColourBacc", bare_num(s["colour_balanced_accuracy"])),
                ("ColourAurocMedian", bare_num(s["colour_auroc_median"])),
                ("CromaGivenColourMedian", bare_num(s["croma_given_colour_median"])),
                ("NGivenColourAboveHalf", str(s["n_given_colour_above_half"])),
                ("ColourRhoMedian", num_math(s["rho_colour_median"])),
            ]
        out += [rf"\newcommand{{\{p}{name}}}{{{body}}}" for name, body in specs]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    rendered = build()
    if not out.parent.exists():
        print(f"skip   {out} (paper/ absent)", file=sys.stderr)
        return 0
    out.write_text(rendered)
    print(f"wrote  {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
