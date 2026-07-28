"""Export the published benchmark results from ``output/`` into the tracked ``results/`` tree.

The documentation site publishes numbers. It builds with ``sphinx -W`` on a clean CI
checkout, which can see neither ``output/`` (git-ignored) nor ``scripts/repro/``
(git-ignored, ADR-0012), so every number the site shows must already be committed. This
script is the only way numbers get there: it reads each published cohort's run and writes
a small set of CSVs, a binned distribution payload for the explorer, and a provenance
sidecar naming the run and checksumming every file it wrote. See ADR-0016.

Run from the repository root::

    python scripts/tools/export_results.py            # rewrite results/
    python scripts/tools/export_results.py --check    # report drift, write nothing

``--check`` is what ``tests/test_results_export.py`` drives, so a benchmark re-run that
was never republished fails a test rather than leaving the public site asserting numbers
no run produced.

This deliberately does *not* share code with the paper's table generators. They live in
the git-ignored ``scripts/repro/`` tree, and importing across that boundary would make a
public artifact depend on a private one. The two paths render overlapping numbers and are
not unified; ADR-0016 records the trade.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# `_pareto_frontier_max_max` is the primitive the paper's Pareto figure rings. Reusing it
# means the site's frontier column and the manuscript's figure cannot disagree about which
# encoders are undominated. It lives in the tracked benchmark plot library.
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from plotting.base import _pareto_frontier_max_max  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from croma import __version__ as CROMA_VERSION  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402

#: The protocol every published cohort is reported at. All three share one operating point
#: per cohort (the dataset median of the per-model biological k*), which is what makes a
#: cross-cohort rank meaningful.
PROTOCOL = "median-k"

#: Resolution of the committed distribution payload. 200 bins resolves the lower decile
#: that LTM10 summarises while keeping the whole payload under a hundred kilobytes.
N_BINS = 200

#: Decimals kept in the published CSVs. Six is well past anything the site renders (the
#: tables print two or three) and past the run-to-run reproducibility of a k-NN sweep, but
#: it keeps the CSVs usable as data rather than as a screenshot of a table.
VALUE_PRECISION = 6

#: Mean ranks are means of small integers, so they land on thirds. Three decimals is exact
#: for a three-cohort mean and does not pretend to more.
RANK_PRECISION = 3


@dataclass(frozen=True)
class Cohort:
    """One published cohort: where its run lives and what the site calls it."""

    slug: str
    benchmark: str
    label: str

    @property
    def run_dir(self) -> Path:
        return ROOT / "output" / "metrics" / PROTOCOL / self.benchmark

    @property
    def metrics_csv(self) -> Path:
        return self.run_dir / "results" / "metrics.csv"

    @property
    def per_sample_csv(self) -> Path:
        return self.run_dir / "results" / "per_sample_metrics.csv"


#: The three published cohorts. Prostate (16-model roster) and PANDA (slide-level, n=4)
#: are computed and stay in ``output/``: a site table whose roster silently differs from
#: the one beside it misleads more than it informs. TCGA-2x2 is excluded for the same
#: reason its results table is a supplement -- it is the same corpus as TCGA-4x4 at a
#: coarser confounder split, and two near-duplicate cohorts crowd the aggregate.
COHORTS: tuple[Cohort, ...] = (
    Cohort("camelyon", "pathorob-camelyon", "Camelyon"),
    Cohort("tcga-4x4", "pathorob-tcga-4x4", "TCGA-4×4"),
    Cohort("tolkach-esca", "pathorob-tolkach-esca", "Tolkach-ESCA"),
)

#: Published columns, in order, for a per-cohort table. Matches the manuscript's results
#: table: two retrieval diagnostics, the two pooled counts and their difference, then the
#: three distributional statistics and the support the counts rest on.
COHORT_COLUMNS = [
    "model",
    "is_control",
    "bio_bacc",
    "conf_bacc",
    "ri",
    "mari",
    "delta",
    "croma",
    "croma_f0",
    "croma_ltm10",
    "support",
]


def _cohort_column(slug: str) -> str:
    return "croma_" + slug.replace("-", "_")


#: Published columns, in order, for the aggregate. The two ranks come before the values
#: they summarise, because the ranks are the reason the table exists.
def aggregate_columns(slugs) -> list[str]:
    return ["model", "is_control", "on_frontier", "croma_rank", "ltm_rank"] + [
        _cohort_column(slug) for slug in slugs
    ]


# --------------------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------------------


def croma_as_margin(croma: pd.Series) -> pd.Series:
    """Map a stored CRoMa column onto the canonical margin scale.

    CRoMa is a ratio of same-confounder to other-confounder margin evidence, and runs from
    different eras of this repository stored either the ratio (neutral at 1) or the margin
    ``(r-1)/(r+1)`` (neutral at 0). Publishing whichever happened to be on disk would put a
    ``1.50`` in one cell and a ``0.20`` in the next. A negative value is decisive for
    margin, a value above 1 for ratio; an all-``[0, 1]`` column is ambiguous and defaults to
    margin, which is the scale the manuscript reports.
    """
    croma = croma.astype(float)
    if (croma < 0.0).any():
        return croma
    if (croma > 1.0 + 1e-9).any():
        return (croma - 1.0) / (croma + 1.0)
    return croma


# --------------------------------------------------------------------------------------
# Table assembly. Pure functions over frames, so the tests can drive them with synthetic
# data and never need a benchmark run.
# --------------------------------------------------------------------------------------


def build_cohort_table(metrics: pd.DataFrame, frac_neg: dict[str, float]) -> pd.DataFrame:
    """One cohort's published table, best CRoMa first.

    ``frac_neg`` maps model to F(0), the fraction of samples whose CRoMa is negative --
    the one published column that cannot be read off ``metrics.csv``, because it is a
    property of the per-sample distribution rather than of its summary.

    The natural-image control is sorted inline rather than banded off beneath a rule: on
    these three cohorts it lands mid-panel, and hiding that would misrepresent what the
    floor actually is. The ``is_control`` flag is what the site's footnote hangs on.
    """
    out = pd.DataFrame(
        {
            "model": metrics["model"].astype(str),
            "is_control": metrics["model"].astype(str) == CONTROL_MODEL,
            "bio_bacc": metrics["bio_knn_bacc"].astype(float),
            "conf_bacc": metrics["confounder_knn_bacc"].astype(float),
            "ri": metrics["ri"].astype(float),
            "mari": metrics["mari"].astype(float),
            "croma": croma_as_margin(metrics["croma"]),
            "croma_ltm10": metrics["croma_ltm_alpha"].astype(float),
            # Support is the fraction of samples that actually contribute to the counts.
            # RI and MaRI share a neighbourhood, so they share an undefined set.
            "support": 1.0 - metrics["ri_undefined_frac"].astype(float),
        }
    )
    out["delta"] = out["mari"] - out["ri"]
    missing = sorted(set(out["model"]) - set(frac_neg))
    if missing:
        raise KeyError(f"no F(0) for {missing}; the per-sample distribution is incomplete")
    out["croma_f0"] = out["model"].map(frac_neg).astype(float)
    return (
        out[COHORT_COLUMNS]
        .sort_values("croma", ascending=False, kind="stable")
        .reset_index(drop=True)
        .round(VALUE_PRECISION)
    )


def build_aggregate_table(per_cohort: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The cross-cohort aggregate: two mean ranks and the three margins behind them.

    Each model is ranked within every cohort by median CRoMa (1 = highest margin) and by
    LTM10 (1 = mildest tail), and the published ranks are the means of those across
    cohorts. There is deliberately **no combined rank**: the mean of a margin rank and a
    tail rank is exactly the composite scalar the manuscript's two-axis framing exists to
    refuse, and the site must not quietly elect a winner the paper declines to.

    ``on_frontier`` marks the encoders no other encoder beats on both axes at once. That
    is the honest summary of a two-axis comparison -- a set, not an order.

    Only models present in every cohort take part; a model scored on two of three has no
    comparable mean rank.
    """
    croma = pd.DataFrame({slug: df.set_index("model")["croma"] for slug, df in per_cohort.items()})
    ltm = pd.DataFrame(
        {slug: df.set_index("model")["croma_ltm10"] for slug, df in per_cohort.items()}
    )
    complete = croma.dropna().index.intersection(ltm.dropna().index)
    croma, ltm = croma.loc[complete], ltm.loc[complete]

    # `method="first"` breaks exact ties by row order rather than averaging them into a
    # fractional rank, so a rank is always an integer before it is averaged across cohorts.
    croma_rank = croma.rank(ascending=False, method="first").mean(axis=1)
    ltm_rank = ltm.rank(ascending=False, method="first").mean(axis=1)

    # Negated: the primitive prefers larger on both axes, and a smaller mean rank is better.
    frontier = set(
        _pareto_frontier_max_max(
            [(m, -float(croma_rank[m]), -float(ltm_rank[m])) for m in complete]
        )
    )

    out = pd.DataFrame(
        {
            "model": list(complete),
            "is_control": [m == CONTROL_MODEL for m in complete],
            "on_frontier": [m in frontier for m in complete],
            "croma_rank": croma_rank.to_numpy(),
            "ltm_rank": ltm_rank.to_numpy(),
        }
    )
    for slug in per_cohort:
        out[_cohort_column(slug)] = croma[slug].to_numpy()
    out[["croma_rank", "ltm_rank"]] = out[["croma_rank", "ltm_rank"]].round(RANK_PRECISION)
    return (
        out[aggregate_columns(per_cohort)]
        .sort_values(["croma_rank", "ltm_rank", "model"], kind="stable")
        .reset_index(drop=True)
        .round(VALUE_PRECISION)
    )


def bin_distribution(values, lo: float, hi: float, n_bins: int = N_BINS) -> list[int]:
    """Histogram ``values`` over ``n_bins`` equal bins spanning ``[lo, hi]``.

    Values outside the range fall into the nearest edge bin rather than being dropped, so
    the counts always sum to ``len(values)``. The explorer reports "how many samples fall
    in this range" and a histogram that silently loses tail mass would answer wrongly --
    which is precisely the question the tail statistics exist to make askable.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if not hi > lo:
        raise ValueError(f"empty range [{lo}, {hi}]")
    series = pd.Series(values, dtype=float).dropna()
    width = (hi - lo) / n_bins
    idx = ((series - lo) / width).astype(int).clip(0, n_bins - 1)
    return idx.value_counts().reindex(range(n_bins), fill_value=0).sort_index().tolist()


def build_distributions(per_sample: dict[str, pd.DataFrame]) -> dict:
    """The explorer payload: per-cohort, per-model binned CRoMa at the headline radius.

    Each cohort gets its own bin grid, spanning that cohort's observed range rounded
    outward to two decimals. A shared grid across cohorts would spend most of its
    resolution on empty space, and the lower decile is the part that has to stay readable.
    """
    payload = {"m": int(CROMA_HEADLINE_M), "n_bins": N_BINS, "cohorts": {}}
    for cohort in COHORTS:
        frame = per_sample[cohort.slug]
        column = f"croma_m{int(CROMA_HEADLINE_M)}"
        lo = float(pd.Series(frame[column]).min())
        hi = float(pd.Series(frame[column]).max())
        lo, hi = _round_outward(lo, hi)
        payload["cohorts"][cohort.slug] = {
            "label": cohort.label,
            "lo": lo,
            "hi": hi,
            "models": {
                str(model): bin_distribution(group[column], lo, hi)
                for model, group in frame.groupby("model", sort=True)
            },
        }
    return payload


def _round_outward(lo: float, hi: float, places: int = 2) -> tuple[float, float]:
    """Widen ``[lo, hi]`` to the enclosing grid at ``places`` decimals."""
    step = 10.0**-places
    return (
        round(float(f"{lo - step:.{places}f}"), places),
        round(float(f"{hi + step:.{places}f}"), places),
    )


# --------------------------------------------------------------------------------------
# Reading a run
# --------------------------------------------------------------------------------------


def load_frac_neg(per_sample: pd.DataFrame) -> dict[str, float]:
    """Per-model F(0) from a cohort's per-sample frame."""
    column = f"croma_m{int(CROMA_HEADLINE_M)}"
    return per_sample.groupby("model")[column].apply(lambda s: float((s < 0).mean())).to_dict()


def read_per_sample(cohort: Cohort) -> pd.DataFrame:
    """The two per-sample columns the export needs.

    Read narrowly on purpose: the full per-sample frame runs to hundreds of megabytes per
    cohort, and both the F(0) column and the distribution payload need only the model name
    and its CRoMa at the headline radius.
    """
    return pd.read_csv(cohort.per_sample_csv, usecols=["model", f"croma_m{int(CROMA_HEADLINE_M)}"])


def export(cohorts: tuple[Cohort, ...] = COHORTS) -> dict[str, str]:
    """Render every published artifact in memory. Returns path -> file content.

    Nothing is written here, so ``--check`` and the freshness test compare exactly what a
    write would have produced.
    """
    tables: dict[str, pd.DataFrame] = {}
    per_sample: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}

    for cohort in cohorts:
        metrics = pd.read_csv(cohort.metrics_csv)
        samples = read_per_sample(cohort)
        per_sample[cohort.slug] = samples
        tables[cohort.slug] = build_cohort_table(metrics, load_frac_neg(samples))
        meta[cohort.slug] = _cohort_provenance(cohort, metrics)

    aggregate = build_aggregate_table(tables)
    distributions = build_distributions(per_sample)

    rendered = {f"results/{slug}.csv": _to_csv(df) for slug, df in tables.items()}
    rendered["results/cross_benchmark.csv"] = _to_csv(aggregate)
    rendered["results/distributions.json"] = _to_compact_json(distributions)
    rendered["results/PROVENANCE.json"] = _to_json(_provenance(meta, rendered, aggregate))
    rendered["README.md"] = render_readme(aggregate, meta)
    return rendered


# --------------------------------------------------------------------------------------
# The README's table
# --------------------------------------------------------------------------------------

#: The README shows the head of the panel rather than all 21. Truncating is a real
#: judgement call -- an encoder at position 9 may well notice -- so the rule is mechanical,
#: stated in the caption, and the full panel is one link away.
README_TOP = 8

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"

DOCS = "https://clemsgrs.github.io/croma"


def render_readme(aggregate: pd.DataFrame, meta: dict[str, dict]) -> str:
    """The README with its results region replaced by the current aggregate.

    A hand-written table in the README is the same hazard as a hand-written one on the
    site, with less to catch it -- so this region is generated too, and the freshness test
    covers the README exactly as it covers the CSVs. Everything outside the markers is left
    alone.
    """
    readme = (ROOT / "README.md").read_text()
    start, end = readme.find(README_START), readme.find(README_END)
    if start < 0 or end < 0:
        raise ValueError(
            f"README.md has no {README_START} / {README_END} region for the results table."
        )
    block = _readme_block(aggregate, meta)
    return readme[: start + len(README_START)] + block + readme[end:]


def _readme_block(aggregate: pd.DataFrame, meta: dict[str, dict]) -> str:
    labels = {_cohort_column(slug): info["label"] for slug, info in meta.items()}
    cohort_columns = [c for c in aggregate.columns if c in labels]
    headers = ["Model", "CRoMa rank", "tail rank"] + [labels[c] for c in cohort_columns]

    lines = [
        "",
        "| " + " | ".join(headers) + " |",
        "| --- |" + " ---: |" * (len(headers) - 1),
    ]
    for _, row in aggregate.head(README_TOP).iterrows():
        name = f"**{row['model']}**" if row["on_frontier"] else str(row["model"])
        if row["is_control"]:
            name += " †"
        cells = [name, f"{row['croma_rank']:.1f}", f"{row['ltm_rank']:.1f}"]
        cells += [f"{row[c]:.2f}" for c in cohort_columns]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"Top {README_TOP} of {len(aggregate)} by CRoMa rank, over {len(cohort_columns)} tile "
        f"cohorts. Each rank is the mean of that encoder's within-cohort ranks — by median "
        f"CRoMa, and by tail severity LTM₁₀. **Bold** marks the Pareto frontier: the encoders "
        f"no other encoder beats on both at once. There is deliberately no combined rank, "
        f"because a strong median can hide a brittle tail.",
        "",
        f"📊 **[Full panel, per-cohort detail and the distributions]({DOCS}/results/)**",
        "",
    ]
    return "\n".join(lines)


def _cohort_provenance(cohort: Cohort, metrics: pd.DataFrame) -> dict:
    ks = sorted({int(k) for k in metrics["k"]})
    if len(ks) != 1:
        raise ValueError(
            f"{cohort.slug}: {PROTOCOL} promises one shared operating point, run has k={ks}. "
            f"A cross-cohort rank over per-model k is not the statistic the site describes."
        )
    return {
        "benchmark": cohort.benchmark,
        "label": cohort.label,
        "run": str(cohort.run_dir.relative_to(ROOT)),
        "k": ks[0],
        "n_models": int(len(metrics)),
        "confounder": str(metrics["confounder_display_name"].iloc[0]),
        "evaluation_design": str(metrics["evaluation_design"].iloc[0]),
    }


def _provenance(meta: dict[str, dict], rendered: dict[str, str], aggregate: pd.DataFrame) -> dict:
    """The sidecar. Everything a reader needs to say which run this table describes."""
    return {
        "croma_version": CROMA_VERSION,
        "exported": dt.date.today().isoformat(),
        "protocol": PROTOCOL,
        "croma_m": int(CROMA_HEADLINE_M),
        "tail_alpha": 0.1,
        # Never a pinned temperature: each model's tau is the median typed-neighbour
        # distance of its own embedding at k, which is the only setting under which MaRI
        # is comparable across models.
        "tau_policy": "auto (per-model median typed-neighbour distance at k)",
        "roster": int(len(aggregate)),
        "cohorts": meta,
        # Every data file this export writes, and nothing else: a file in results/ that is
        # absent here was not produced by this script. Scoped to results/ deliberately --
        # the export also rewrites a region of README.md, but that file is mostly prose and
        # a checksum over it would go stale on every wording change while saying nothing
        # about the numbers. The freshness test covers the README instead.
        "files": {
            path: hashlib.sha256(content.encode()).hexdigest()
            for path, content in sorted(rendered.items())
            if path.startswith("results/") and not path.endswith("PROVENANCE.json")
        },
    }


def _to_csv(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _to_compact_json(payload: dict) -> str:
    """Whitespace-free JSON, for the payload the browser downloads.

    Indenting 12,600 bin counts puts each one on its own line and triples the transfer
    size for no benefit: a diff of histogram counts is not human-reviewable at any
    indentation, and this file is fetched by every visitor to the results page.
    """
    return json.dumps(payload, separators=(",", ":"), sort_keys=False) + "\n"


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift against the committed results/ tree; write nothing. Exit 1 if stale.",
    )
    args = parser.parse_args(argv)

    missing = [c.metrics_csv for c in COHORTS if not c.metrics_csv.exists()]
    if missing:
        for path in missing:
            print(f"missing run: {path}", file=sys.stderr)
        print(
            "\noutput/ is git-ignored, so this only runs on a machine holding the runs.",
            file=sys.stderr,
        )
        return 1

    rendered = export()

    if args.check:
        stale = 0
        for path, content in sorted(rendered.items()):
            target = ROOT / path
            # PROVENANCE carries an export date, so it differs on every run by design.
            # Its checksums are what pin the data, and those are compared through the
            # files they describe.
            if path.endswith("PROVENANCE.json"):
                continue
            current = target.read_text() if target.exists() else ""
            if current != content:
                stale += 1
                print(f"STALE  {path}", file=sys.stderr)
            else:
                print(f"ok     {path}", file=sys.stderr)
        if stale:
            print(
                f"\n{stale} file(s) stale; run python scripts/tools/export_results.py",
                file=sys.stderr,
            )
            return 1
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    for path, content in sorted(rendered.items()):
        target = ROOT / path
        target.write_text(content)
        print(f"wrote  {path}  ({len(content) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
