"""Supplementary table: slide encoders versus mean-pooling of their own tile encoders.

One block per pair in ``_slide_vs_meanpool.RENDERED_PAIRS``: the slide arm, the
mean-pooled arm, and their paired difference (slide minus mean-pool). Values come from
``output/studies/slide-vs-meanpool/`` (``scripts/studies/slide_vs_meanpool.py evaluate``);
a rendered pair without output there fails the build. The caption names each declared pair
left out with the reason the study recorded for not running it.

Run: python scripts/repro/generate_slide_vs_meanpool_table.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from _paper_tables import num_math, scriptsize_ci
from _slide_vs_meanpool import (
    CROMA_STATISTICS,
    DECIMALS,
    NIPD_STATISTICS,
    REPO,
    UNDEFINED_SPREAD,
    Basis,
    load,
    pct,
    tex_text,
    tile_display,
)

sys.path.insert(0, str(REPO / "scripts" / "studies"))
from slide_vs_meanpool import REPRODUCTION_COSINE  # noqa: E402

OUT = Path("paper/sections/supp/slide_vs_meanpool.tex")
LABEL = "tab:slide-vs-meanpool"
ARM_LABEL = {"slide": "slide", "meanpool": "mean-pool"}
PROTOCOL_TEX = {"k-star": r"$k^\star$", "median-k": r"the shared median $k$"}


def _count(n: int) -> str:
    return "$" + f"{int(n):,}".replace(",", "{,}") + "$"


def _series(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _arm_row(basis: Basis, pair: str, arm: str) -> str:
    row = basis.summary.loc[(pair, arm)]
    cells = [
        num_math(row["bio_knn_bacc"], 3),
        num_math(row["croma"], 2),
        num_math(row["croma_f0"], 3),
        num_math(row["croma_ltm_alpha"], 2),
        pct(row["nipd_id"]),
        pct(row["nipd_ood"]),
    ]
    lead = rf"\multirow{{3}}{{*}}{{{pair}}}" if arm == "slide" else ""
    return f"{lead} & {ARM_LABEL[arm]} & " + " & ".join(cells) + r" \\"


def _difference_row(basis: Basis, pair: str) -> str:
    cells = []
    for statistic in CROMA_STATISTICS:
        row = basis.differences.loc[(pair, statistic)]
        d = DECIMALS[statistic]
        cells.append(num_math(row["point"], d) + r"\," + scriptsize_ci(row["lo"], row["hi"], d))
    for statistic in NIPD_STATISTICS:
        row = basis.differences.loc[(pair, statistic)]
        if math.isnan(float(row["spread"])):
            spread = rf"{{\scriptsize$\pm${UNDEFINED_SPREAD}}}"
        else:
            spread = rf"{{\scriptsize$\pm{100.0 * float(row['spread']):.1f}$}}"
        cells.append(pct(row["point"]) + r"\," + spread)
    # The bio bacc cell stays empty: no paired difference is reported for it.
    return r" & $\Delta$ & & " + " & ".join(cells) + r" \\"


def _spread_clause(basis: Basis, iterations: int) -> str:
    """What the nIPD ``±`` is the SD of, and over how many replicates.

    A spread is over the paired replicates on which both arms' nIPD is defined (the
    ``n`` column). The caption gives that count when every comparison shares it, and only
    the rule when they differ; a spread over fewer than two prints as a dash.
    """
    rows = basis.differences.loc[[(p, s) for p in basis.rendered for s in NIPD_STATISTICS]]
    counts = sorted(set(rows["n"].astype(int)))
    rule = r"paired probe replicates in which both arms' \code{nIPD} is defined"
    if counts == [int(iterations)]:
        clause = rf"$\pm$ the standard deviation of the {_count(iterations)} paired probe replicates"
    elif len(counts) == 1:
        clause = rf"$\pm$ the standard deviation of the {_count(counts[0])} of {_count(iterations)} {rule}"
    else:
        clause = rf"$\pm$ the standard deviation over the {rule}"
    if rows["spread"].isna().any():
        clause += f"; {UNDEFINED_SPREAD} where fewer than two are"
    return clause


def _caption(basis: Basis) -> str:
    prov = basis.provenance
    records = {p: prov["pairs"][p]["reproduction"] for p in basis.rendered}
    n_slides = next(iter(records.values()))["n_slides"]
    threshold = next(iter(records.values())).get("threshold", REPRODUCTION_COSINE)
    kept = [p for p in basis.rendered if records[p]["reproduced"]]
    rerun = [p for p in basis.rendered if not records[p]["reproduced"]]
    boot, probe, rep = prov["bootstrap"], prov["downstream"], prov["representation"]
    m = int(rep["croma_headline_m"])
    level = int(round(100 * float(boot["level"])))

    parts = [
        r"\caption{\textbf{Slide encoders versus mean-pooling of their own tile embeddings on "
        r"PCaBiop.} ",
        "Pairs (slide encoder, tile encoder): "
        + _series([f"{p} ({tile_display(p)})" for p in basis.rendered])
        + ". ",
        "Each slide encoder's published slide2vec configuration was re-run on the "
        + _count(n_slides)
        + " PCaBiop slides with its tile embeddings saved, so both arms come from one tiling: "
        "the slide arm is the slide encoder's embedding and the mean-pool arm is the "
        "unweighted mean of the same tile embeddings. ",
    ]
    def reproduced(p: str) -> str:
        r = records[p]
        return (
            f"{_count(r['n_reproduced'])} of {_count(r['n_slides'])} slides reproduced "
            rf"(minimum cosine ${float(r['min_cosine']):.3f}$)"
        )

    criterion = (
        "Re-run slide embeddings reproduced the published ones (per-slide cosine "
        rf"$\geq{threshold:g}$ on every slide) for "
    )
    if kept:
        parts.append(
            criterion + f"{_series(kept)}, whose published embeddings form the slide arm. "
        )
        for p in rerun:
            parts.append(
                f"They did not for {p}: {reproduced(p)}, so its re-run slide embeddings "
                "form the slide arm. "
            )
    else:
        # No sentence about reproduced pairs to refer back to: state the criterion once,
        # then give every pair's count.
        parts.append(
            criterion
            + "none of the pairs, so each pair's re-run slide embeddings form its slide arm: "
            + "; ".join(f"{p}, {reproduced(p)}" for p in rerun)
            + ". "
        )
    parts += [
        rf"bio bacc, median \code{{CRoMa}}$(m{{=}}{m})$, $F(0)$ and $\mcode{{LTM}}_{{10}}$ are "
        "from the benchmark runner on the in-domain PANDA cohort at "
        + PROTOCOL_TEX.get(rep["protocol"], rep["protocol"])
        + r"; \code{nIPD} (\%) is from the PCaBiop linear-probe sweep, in domain (PANDA) and "
        "out of domain (PAR). ",
        r"$\Delta$ is slide minus mean-pool: \code{CRoMa} summaries with "
        rf"{level}\% confidence intervals from a stratified slide bootstrap (class $\times$ "
        f"provider strata, {_count(boot['n_boot'])} replicates, one draw shared by every arm), "
        r"and \code{nIPD} differences (" + _spread_clause(basis, probe["iterations"]) + "). ",
    ]
    # ``load`` has checked that every declared pair left out is optional and recorded as
    # not run, with a reason.
    for p in basis.missing:
        parts.append(
            f"{p} ({tile_display(p)}), an optional pair, was not run: "
            f"{tex_text(basis.not_run[p].strip())}. "
        )
    return "".join(parts).rstrip() + "}"


def build(basis: Basis) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        r"Slide encoder & Arm & bio bacc & \code{CRoMa} & $F(0)$ & $\mcode{LTM}_{10}$ & "
        r"\code{nIPD}$_\mathrm{ID}$ & \code{nIPD}$_\mathrm{OOD}$ \\",
        r"\hline",
    ]
    for pair in basis.rendered:
        lines.append(_arm_row(basis, pair, "slide"))
        lines.append(_arm_row(basis, pair, "meanpool"))
        lines.append(_difference_row(basis, pair))
        lines.append(r"\hline")
    lines += [r"\end{tabular}", _caption(basis), rf"\label{{{LABEL}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    args.out.write_text(build(load(args.study_dir)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
