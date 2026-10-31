r"""Build ``tab:distance-variants``: CRoMa under alternative typed-neighbour distances.

Panel (a) compares each alternative geometry with the cosine reference on every tile
benchmark -- ranking agreement for median CRoMa and LTM10, sign changes of the median, and
the medians over encoders of |delta F(0)| and per-sample sign agreement. Panel (b) gives
rho(median CRoMa, nIPD) under every geometry, cosine's first, on the three cohorts with
downstream cells. A footer reports the retained whitening dimensions and Camelyon's 90% /
99% whitening sensitivity, and the caption states how the cosine reference reproduced each
run: exactly, or within the benchmark's tolerance with the largest deviation. ``macros``
emits the numbers the text may quote; they carry numbers, not verdicts.

Every number is read from the study (``_distance_variants.load``), which refuses a missing,
incomplete or stale study; nothing here computes a statistic, so a rebuild is
byte-for-byte.

Run: python scripts/repro/generate_distance_variants_table.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _distance_variants as dvs  # noqa: E402
from _paper_tables import (  # noqa: E402
    CaptionClaimError,
    bare_num,
    num_math,
    sci_math,
    scriptsize_ci,
)
from paper_manifest import by_benchmark  # noqa: E402

OUT = Path("paper/sections/supp/distance_variants.tex")
LABEL = "tab:distance-variants"

#: Display name and macro suffix per variant.
VARIANT_NAMES = {
    "cosine": ("Cosine (reference)", "Cosine"),
    "normalised-euclidean": ("Normalized Euclidean", "NormalizedEuclidean"),
    "euclidean": ("Raw Euclidean", "RawEuclidean"),
    "whitened": ("PCA-whitened", "Whitened"),
}
NIPD_COLUMNS = (("nipd_id", r"\code{nIPD}$_\mathrm{ID}$"), ("nipd_ood", r"\code{nIPD}$_\mathrm{OOD}$"))
P_FLOOR = 0.001


def _rho_ci(row, prefix: str) -> str:
    return f"{num_math(float(row[prefix]))} {scriptsize_ci(float(row[prefix + '_lo']), float(row[prefix + '_hi']))}"


def _p(value: float) -> str:
    return rf"${{<}}{P_FLOOR:g}$" if value < P_FLOOR else f"${value:.3f}$"


def _pct(value: float) -> str:
    return rf"${100.0 * value:.1f}\%$"


def _int(value) -> int:
    as_float = float(value)
    if as_float != int(as_float):
        raise CaptionClaimError(f"expected a count, got {value}")
    return int(as_float)


def _dims(row) -> str:
    """Retained whitening dimensions: median (range) over the ranked encoders."""
    median = float(row["dims_median"])
    text = f"{median:.0f}" if median.is_integer() else f"{median:.1f}"
    return f"${text}$ (${_int(row['dims_min'])}$--${_int(row['dims_max'])}$)"


def _thousands(n: int) -> str:
    return f"${int(n):,}$".replace(",", "{,}")


def _n_encoders(data: dvs.DistanceVariants) -> int:
    counts = {int(n) for n in data.measures["n_encoders"]}
    if len(counts) != 1:
        raise CaptionClaimError(f"benchmarks rank different numbers of encoders: {counts}")
    return counts.pop()


def _agreement_rows(data: dvs.DistanceVariants) -> list[str]:
    lines = []
    for benchmark in dvs.BENCHMARK_ORDER:
        name = by_benchmark(benchmark).short_name
        for i, variant in enumerate(dvs.ALTERNATIVES):
            row = data.row(benchmark, variant)
            head = rf"\multirow{{{len(dvs.ALTERNATIVES)}}}{{*}}{{{name}}}" if i == 0 else ""
            cells = [
                head,
                VARIANT_NAMES[variant][0],
                _rho_ci(row, "rank_rho_median"),
                _rho_ci(row, "rank_rho_ltm10"),
                f"${_int(row['sign_changes'])}$",
                num_math(float(row["median_abs_delta_f0"]), 3),
                _pct(float(row["median_sign_agreement"])),
            ]
            lines.append("\t\t" + " & ".join(cells) + r" \\")
        lines.append("\t\t" + r"\hline")
    return lines


def _nipd_rows(data: dvs.DistanceVariants) -> list[str]:
    lines = []
    variants = ("cosine", *dvs.ALTERNATIVES)
    for benchmark in data.nipd_benchmarks():
        name = by_benchmark(benchmark).short_name
        for i, variant in enumerate(variants):
            row = data.row(benchmark, variant)
            head = rf"\multirow{{{len(variants)}}}{{*}}{{{name}}}" if i == 0 else ""
            cells = [head, VARIANT_NAMES[variant][0]]
            for target, _ in NIPD_COLUMNS:
                cells += [_rho_ci(row, f"rho_{target}"), _p(float(row[f"rho_{target}_p"]))]
            lines.append("\t\t" + " & ".join(cells) + r" \\")
        lines.append("\t\t" + r"\hline")
    return lines


def _footer(data: dvs.DistanceVariants) -> str:
    dims = "; ".join(
        f"{by_benchmark(b).short_name} {_dims(data.row(b, 'whitened'))}"
        for b in dvs.BENCHMARK_ORDER
    )
    sensitivity = []
    for variant, share in zip(dvs.SENSITIVITY, ("90", "99")):
        row = data.row("pathorob-camelyon", variant)
        sensitivity.append(
            rf"at ${share}\%$, {_dims(row)} dimensions, ranking $\rho$ "
            rf"{_rho_ci(row, 'rank_rho_median')} for the median and "
            rf"{_rho_ci(row, 'rank_rho_ltm10')} for $\mcode{{LTM}}_{{10}}$, "
            rf"${_int(row['sign_changes'])}$ sign changes"
        )
    return (
        r"Retained whitening dimensions at $95\%$ of the variance, median (range) over "
        rf"encoders: {dims}. Camelyon whitened {sensitivity[0]}; {sensitivity[1]}."
    )


def _join(items: list[str], last: str = " and ") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + last + items[-1]


def _encoders(n: int) -> str:
    return "one encoder" if n == 1 else f"${n}$ encoders"


def _reproduction_sentence(data: dvs.DistanceVariants) -> str:
    """One sentence, from the study's provenance: on which benchmarks the cosine reference
    reproduces the runs' per-sample values exactly, and on which only within its tolerance,
    with the largest deviation and how many encoders no kernel reproduces exactly."""
    within = [b for b in dvs.BENCHMARK_ORDER if not data.reproduction(b)["identical"]]
    if not within:
        return "The cosine reference reproduces every run's per-sample values exactly."
    exact = [by_benchmark(b).short_name for b in dvs.BENCHMARK_ORDER if b not in within]
    clauses = [f"exactly on {_join(exact)}"] if exact else []
    for b in within:
        record = data.reproduction(b)
        clauses.append(
            f"on {by_benchmark(b).short_name} only within an absolute tolerance of "
            f"{sci_math(float(record['tolerance']))} (largest deviation "
            f"{sci_math(float(record['max_deviation']))}, for "
            f"{_encoders(len(record['within_tolerance']))} that no BLAS kernel reproduces "
            f"exactly)"
        )
    return (
        f"The cosine reference reproduces the runs' per-sample values "
        f"{_join(clauses, ', and ')}."
    )


def build(data: dvs.DistanceVariants | None = None) -> str:
    data = dvs.load() if data is None else data
    n = _n_encoders(data)
    outputs = data.provenance["outputs"]
    lines = [
        "% AUTO-GENERATED by scripts/repro/generate_distance_variants_table.py -- do not edit by hand.",
        "% Read from output/studies/distance-variants/: "
        + ", ".join(f"{name} sha256 {digest}" for name, digest in sorted(outputs.items())),
        r"\begin{table}[!htbp]",
        "\t" + r"\centering",
        "\t" + r"\footnotesize",
        "\t" + r"\setlength{\tabcolsep}{3pt}",
        "\t" + r"\begin{tabular}{llccccc}",
        "\t\t" + r"\hline",
        "\t\t" + r"\multicolumn{7}{l}{\textbf{a}\quad Agreement with the cosine reference} \\",
        "\t\t" + r"\hline",
        "\t\t"
        + r"& & \multicolumn{2}{c}{Ranking $\rho$} & Median & Median & Per-sample \\",
        "\t\t" + r"\cline{3-4}",
        "\t\t"
        + r"Benchmark & Distance & median \code{CRoMa} & $\mcode{LTM}_{10}$ & sign changes "
        + r"& $|\Delta F(0)|$ & sign agreement \\",
        "\t\t" + r"\hline",
        *_agreement_rows(data),
        "\t" + r"\end{tabular}",
        "",
        "\t" + r"\vspace{0.8em}",
        "\t" + r"\begin{tabular}{llcccc}",
        "\t\t" + r"\hline",
        "\t\t"
        + r"\multicolumn{6}{l}{\textbf{b}\quad Association of median \code{CRoMa} with "
        + r"downstream shortcut susceptibility} \\",
        "\t\t" + r"\hline",
        "\t\t"
        + " & ".join(
            ["&"] + [rf"\multicolumn{{2}}{{c}}{{{label}}}" for _, label in NIPD_COLUMNS]
        )
        + r" \\",
        "\t\t" + r"\cline{3-4}\cline{5-6}",
        "\t\t" + r"Benchmark & Distance & $\rho$ & $p$ & $\rho$ & $p$ \\",
        "\t\t" + r"\hline",
        *_nipd_rows(data),
        "\t" + r"\end{tabular}",
        "",
        "\t" + r"\vspace{0.5em}",
        "\t" + r"\parbox{\linewidth}{\scriptsize " + _footer(data) + "}",
        "\t"
        + r"\caption{\textbf{\code{CRoMa} under alternative typed-neighbour distances.} "
        + r"Every tile benchmark is re-scored at $m=5$, with its own evaluation design and "
        + r"neighbour search, under three geometries besides the cosine reference, the run's "
        + r"own computation: Euclidean distance on L2-normalized features, Euclidean "
        + r"distance on the raw features, and Euclidean "
        + r"distance after PCA whitening, which keeps the fewest leading components "
        + r"explaining $95\%$ of the variance of the evaluated samples and scales each to "
        + r"unit variance (a Mahalanobis distance under the truncated covariance; no labels "
        + r"or centres are used). "
        + _reproduction_sentence(data)
        + r" (a) Agreement with cosine over the "
        + rf"${n}$ ranked encoders of each benchmark: Spearman $\rho$ between the encoder "
        + r"rankings by median \code{CRoMa} and by $\mcode{LTM}_{10}$, the number of encoders "
        + r"whose median changes sign (a median $\leq 0$ counts as confounder-dominant, as in "
        + r"$F(0)$), and the medians over encoders of the absolute change in $F(0)$ and of the "
        + r"share of evaluation units whose per-sample sign agrees with cosine. (b) Spearman "
        + r"$\rho$ between median \code{CRoMa} under each geometry and \code{nIPD} on the "
        + r"three cohorts with downstream cells, cosine's first; two-sided $p$ from "
        + _thousands(dvs.study.N_PERMUTATIONS)
        + r" permutations. Brackets are $95\%$ percentile intervals over "
        + _thousands(dvs.study.N_BOOT)
        + r" encoder resamples, one draw shared by every correlation of a benchmark. "
        + r"The natural-image control is excluded throughout.}",
        "\t" + rf"\label{{{LABEL}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def macros(root: Path = dvs.REPO) -> list[str]:
    """The macros of the study under ``root``; a missing or stale study raises, never skips."""
    root = Path(root)
    return macro_lines(dvs.load(root / dvs.STUDY_DIR.relative_to(dvs.REPO), repo=root))


def macro_lines(data: dvs.DistanceVariants) -> list[str]:
    """Numbers the text may quote: they carry numbers, not verdicts."""
    lines = ["% CRoMa under alternative distances (tab:distance-variants)"]
    for variant in dvs.ALTERNATIVES:
        suffix = VARIANT_NAMES[variant][1]
        rows = [data.row(b, variant) for b in dvs.BENCHMARK_ORDER]
        median_rho = min(float(r["rank_rho_median"]) for r in rows)
        tail_rho = min(float(r["rank_rho_ltm10"]) for r in rows)
        lines.append(rf"\newcommand{{\DistanceMinRankRho{suffix}}}{{{bare_num(median_rho)}}}")
        lines.append(rf"\newcommand{{\DistanceMinTailRankRho{suffix}}}{{{bare_num(tail_rho)}}}")
    total = sum(
        _int(data.row(b, v)["sign_changes"]) for b in dvs.BENCHMARK_ORDER for v in dvs.ALTERNATIVES
    )
    lines.append(rf"\newcommand{{\DistanceSignChanges}}{{{total}}}")
    for benchmark in data.nipd_benchmarks():
        cohort = by_benchmark(benchmark).prefix
        for variant in ("cosine", *dvs.ALTERNATIVES):
            rho = float(data.row(benchmark, variant)["rho_nipd_id"])
            suffix = VARIANT_NAMES[variant][1]
            lines.append(rf"\newcommand{{\DistanceNipdIdRho{suffix}{cohort}}}{{{bare_num(rho)}}}")
    # The absolute tolerance each benchmark's cosine reproduction was checked within (0:
    # exactly) and the largest per-unit deviation from its run it recorded.
    for benchmark in dvs.BENCHMARK_ORDER:
        cohort = by_benchmark(benchmark).prefix
        record = data.reproduction(benchmark)
        lines += [
            rf"\newcommand{{\DistanceReproductionTolerance{cohort}}}"
            rf"{{{sci_math(float(record['tolerance']))}}}",
            rf"\newcommand{{\DistanceReproductionMaxDeviation{cohort}}}"
            rf"{{{sci_math(float(record['max_deviation']))}}}",
        ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rendered = build()
    if not args.out.parent.exists():
        print(f"skip   {args.out} (paper/ absent)", file=sys.stderr)
        return 0
    args.out.write_text(rendered)
    print(f"wrote  {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
