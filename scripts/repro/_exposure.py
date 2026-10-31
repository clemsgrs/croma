"""Pretraining exposure: the exposure matrix and the drop-exposed sensitivity.

Several encoders were pretrained on data a benchmark scores. The paper flags this with a
dagger per exposed encoder (the Pareto figures and the results tables); this module lays the
extent out in one place (``tab:exposure-matrix``) and recomputes the headline associations and
frontiers with the exposed encoders removed (``tab:drop-exposed``). Overlap stays flagged, not
adjusted.

Every exposure fact is read through ``_model_provenance.exposed_models_for_domain``, the
resolver the daggers use (ADR-0005), so the matrix, the daggers and the drop sets cannot
disagree. A matrix cell is one of three values, and none is inferred:

* ``exposed`` -- the encoder's disclosed corpus includes the dataset (the resolver says so);
* ``not exposed`` -- its disclosed corpus excludes the dataset;
* ``undisclosed`` -- its authors have not disclosed the corpus (``disclosed`` is ``no``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (HERE, REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _model_provenance import exposed_models_for_domain  # noqa: E402
from _paper_tables import CaptionClaimError, bare_num  # noqa: E402
from croma.metrics.bootstrap import (  # noqa: E402
    EXACT_PERMUTATION_MAX_N,
    CorrelationInference,
    spearman_inference,
)
from plotting.style import CONTROL_MODEL, published_models  # noqa: E402

#: The single model registry (ADR-0005), shared with the daggers' resolver.
METADATA = REPO / "scripts" / "bench" / "model_metadata.csv"

EXPOSED = "exposed"
NOT_EXPOSED = "not exposed"
UNDISCLOSED = "undisclosed"


@dataclass(frozen=True)
class SourceDataset:
    """A source dataset some benchmark scores, and the corpus-domain token naming it."""

    domain: str


SOURCE_DATASETS: tuple[SourceDataset, ...] = (
    SourceDataset("tcga"),
    SourceDataset("camelyon"),
    SourceDataset("panda"),
    SourceDataset("par"),
    SourceDataset("tolkach-esca"),
)

#: The source dataset each sensitivity benchmark scores (the manifest benchmark name). The
#: Tolkach-ESCA score excludes its TCGA cohort, so its scored dataset is Tolkach-ESCA itself.
SCORED_DATASET: dict[str, str] = {
    "pathorob-camelyon": "camelyon",
    "pathorob-tcga-2x2": "tcga",
    "pathorob-tcga-4x4": "tcga",
    "pathorob-tolkach-esca": "tolkach-esca",
}


@dataclass(frozen=True)
class Institution:
    """An institution behind one or more scored benchmark centres.

    ``centres`` pairs a manifest benchmark with the confounder label its view gives the
    centre, so a test can check each label against the view itself.
    """

    name: str
    centres: tuple[tuple[str, str], ...]


#: Institutional-domain tokens (``institutional_domains`` in model_metadata.csv) and the scored
#: benchmark centres each one runs. A token that a corpus carries but no entry maps fails the
#: build: an unmapped institution would be a check silently not made.
INSTITUTIONS: dict[str, Institution] = {
    "rumc": Institution(
        "Radboudumc",
        (("pathorob-camelyon", "RUMC"), ("prostate", "RUMC"), ("panda", "radboud")),
    ),
    "charite": Institution(r"Charit\'e", (("pathorob-tolkach-esca", "VALSET4_CHA_FULL"),)),
    "mskcc": Institution(
        "MSKCC", (("pathorob-tcga-2x2", "MSKCC"), ("pathorob-tcga-2x2", "Memorial Sloan Kettering"))
    ),
}


@dataclass(frozen=True)
class InstitutionMark:
    """A disclosed corpus drawn from the institution behind some scored centres.

    Possible source-domain overlap only (CONTEXT.md, "Institutional/source-domain exposure"):
    it is not dataset exposure, and it does not establish that any scored slide was seen.
    ``source`` is the citation key of the disclosure.
    """

    institution: str
    centres: tuple[tuple[str, str], ...]
    source: str


#: Wording for each ``benchmark_selection`` token (ADR-0020). A disclosure of a different kind
#: needs its own wording before it is published, so an unknown token fails the build.
SELECTION_LABELS = {"pathorob-ri": "PathoROB RI"}


@dataclass(frozen=True)
class Selection:
    """A benchmark its authors disclose using during the encoder's development, and the
    citation key of that disclosure. Flagged, never adjusted (ADR-0020)."""

    label: str
    source: str


@dataclass(frozen=True)
class Count:
    """One dataset column of one panel: how many encoders are exposed, how many undisclosed."""

    exposed: int
    undisclosed: int
    n: int


@dataclass(frozen=True)
class Row:
    """One encoder row of the matrix. ``panel`` is ``tile``, ``slide`` or ``control``."""

    model: str
    panel: str


@dataclass(frozen=True)
class ExposureMatrix:
    """One row per encoder, one cell per source dataset."""

    rows: tuple[Row, ...]
    cells: dict[tuple[str, str], str]
    marks: dict[str, tuple[InstitutionMark, ...]]
    selections: dict[str, Selection]

    def status(self, model: str, domain: str) -> str:
        return self.cells[(model, domain)]

    def institutions(self, model: str) -> tuple[InstitutionMark, ...]:
        return self.marks.get(model, ())

    def selection(self, model: str) -> Selection | None:
        return self.selections.get(model)

    def flagged(self, domain: str, roster: set[str]) -> set[str]:
        """The roster members a dagger for ``domain`` marks: exposed in that dataset column, or
        carrying that institution's marker. The domain is a dataset or an institution token."""
        if domain in INSTITUTIONS:
            return {m for m in roster if any(k.institution == domain for k in self.institutions(m))}
        return {m for m in roster if self.status(m, domain) == EXPOSED}

    def counts(self) -> dict[str, dict[str, "Count"]]:
        """Exposed and undisclosed encoders per dataset column, per panel; the control is
        a floor, never counted."""
        out: dict[str, dict[str, Count]] = {}
        for panel in ("tile", "slide"):
            models = [row.model for row in self.rows if row.panel == panel]
            out[panel] = {
                d.domain: Count(
                    exposed=sum(self.status(m, d.domain) == EXPOSED for m in models),
                    undisclosed=sum(self.status(m, d.domain) == UNDISCLOSED for m in models),
                    n=len(models),
                )
                for d in SOURCE_DATASETS
            }
        return out

    def drop_exposed(self, benchmark: str, roster: set[str]) -> set[str]:
        """The encoders the drop-exposed variant removes: exposed to the scored dataset."""
        return self.flagged(SCORED_DATASET[benchmark], roster)

    def disclosed_unexposed(self, benchmark: str, roster: set[str]) -> set[str]:
        """The encoders the drop-exposed-and-undisclosed variant keeps."""
        column = SCORED_DATASET[benchmark]
        return {m for m in roster if self.status(m, column) == NOT_EXPOSED}


def _selections(metadata: pd.DataFrame) -> dict[str, Selection]:
    """The ``benchmark_selection`` column, worded and sourced. Empty means not disclosed."""
    out: dict[str, Selection] = {}
    for _, row in metadata.iterrows():
        model, token = str(row["model"]), str(row["benchmark_selection"]).strip()
        if not token:
            continue
        if token not in SELECTION_LABELS:
            raise ValueError(
                f"{model}: benchmark_selection {token!r} has no wording; add it to "
                "_exposure.SELECTION_LABELS before publishing it (ADR-0020)."
            )
        source = str(row["cite"]).strip()
        if not source:
            raise ValueError(f"{model}: a benchmark_selection disclosure needs a `cite` source.")
        out[model] = Selection(SELECTION_LABELS[token], source)
    return out


def _institution_marks(metadata: pd.DataFrame) -> dict[str, tuple[InstitutionMark, ...]]:
    """Resolve each mapped institution through the dagger resolver, with its source."""
    from _model_provenance import _domains

    models = set(metadata["model"].astype(str))
    tokens = set().union(*(_domains(c) for c in metadata["institutional_domains"]))
    unmapped = sorted(tokens - INSTITUTIONS.keys())
    if unmapped:
        raise ValueError(
            f"institutional domain(s) {unmapped} map to no scored benchmark centre; add them "
            "to _exposure.INSTITUTIONS (with an empty centre list if none is scored)."
        )
    cite = dict(zip(metadata["model"].astype(str), metadata["cite"].astype(str)))
    marks: dict[str, list[InstitutionMark]] = {}
    for token in sorted(tokens):
        centres = INSTITUTIONS[token].centres
        for model in sorted(exposed_models_for_domain(metadata, token, models)):
            source = cite[model].strip()
            if not source:
                raise ValueError(
                    f"{model} shares {INSTITUTIONS[token].name} with a scored centre but has "
                    "no `cite` key; an institution marker must carry its source."
                )
            marks.setdefault(model, []).append(InstitutionMark(token, centres, source))
    return {model: tuple(found) for model, found in marks.items()}


def _check_namespaces(metadata: pd.DataFrame) -> None:
    """Dataset tokens live in ``corpus_domains``, institution tokens in ``institutional_domains``.

    The resolver unions the two columns, so a token in the wrong one would resolve silently as
    the other kind of exposure.
    """
    from _model_provenance import _domains

    datasets = {d.domain for d in SOURCE_DATASETS}
    for _, row in metadata.iterrows():
        wrong = (_domains(row["institutional_domains"]) & datasets) | (
            _domains(row["corpus_domains"]) & INSTITUTIONS.keys()
        )
        if wrong:
            raise ValueError(
                f"{row['model']}: {sorted(wrong)} sits in the wrong provenance column; dataset "
                "tokens belong in corpus_domains and institution tokens in institutional_domains."
            )


def exposure_matrix(metadata: pd.DataFrame) -> ExposureMatrix:
    """Resolve every encoder's exposure to every source dataset."""
    _check_namespaces(metadata)
    models = set(metadata["model"].astype(str))
    disclosed = dict(zip(metadata["model"].astype(str), metadata["disclosed"].astype(str)))
    unknown = sorted(m for m, state in disclosed.items() if state not in ("yes", "no"))
    if unknown:
        raise ValueError(
            f"`disclosed` must be yes or no; {unknown} carry neither, and a cell is never "
            "inferred. Record the disclosure in model_metadata.csv."
        )
    cells: dict[tuple[str, str], str] = {}
    for dataset in SOURCE_DATASETS:
        exposed = exposed_models_for_domain(metadata, dataset.domain, models)
        for model in models:
            if model in exposed:
                cells[(model, dataset.domain)] = EXPOSED
            elif disclosed[model] == "no":
                cells[(model, dataset.domain)] = UNDISCLOSED
            else:
                cells[(model, dataset.domain)] = NOT_EXPOSED
    return ExposureMatrix(
        rows=_rows(metadata),
        cells=cells,
        marks=_institution_marks(metadata),
        selections=_selections(metadata),
    )


def _rows(metadata: pd.DataFrame) -> tuple[Row, ...]:
    """The tile panel, then the slide panel, each in ``panel_order``; the control last."""
    ordered = metadata.sort_values(["panel_order", "model"], kind="stable")
    rows = [
        Row(str(r["model"]), str(r["panel"]))
        for panel in ("tile", "slide")
        for _, r in ordered[ordered["panel"] == panel].iterrows()
        if str(r["model"]) != CONTROL_MODEL
    ]
    rows += [Row(CONTROL_MODEL, "control")] * int((metadata["model"] == CONTROL_MODEL).any())
    return tuple(rows)


# --- rendering --------------------------------------------------------------------------

HEADER = "% AUTO-GENERATED by scripts/repro/generate_exposure_tables.py -- do not edit by hand."

#: Column heads of the dataset columns, in ``SOURCE_DATASETS`` order.
DATASET_HEADS = {
    "tcga": "TCGA",
    "camelyon": r"\shortstack{CAMELYON\\16/17}",
    "panda": "PANDA",
    "par": "PAR",
    "tolkach-esca": r"\shortstack{Tolkach-ESCA\\(UKK, WNS, CHA)}",
}
#: Printed benchmark names where the manifest has no short name.
_BENCHMARK_FALLBACK = {"panda": "PANDA"}
#: Printed centre names where the view's label is not a printable name.
_CENTRE_NAMES = {"VALSET4_CHA_FULL": "CHA"}
_CELL = {EXPOSED: r"\textbf{exposed}", NOT_EXPOSED: "not exposed", UNDISCLOSED: "undisclosed"}
_PANEL_HEADS = {"tile": "Tile encoders", "slide": "Slide encoders"}


def fingerprint_lines(fingerprints: list[tuple[str, str]]) -> list[str]:
    """One ``% input <path> sha256=<hex>`` comment per input the table was built from."""
    return [f"% input {path} sha256={digest}" for path, digest in fingerprints]


def _benchmark_name(benchmark: str) -> str:
    from paper_manifest import by_benchmark

    return by_benchmark(benchmark).short_name or _BENCHMARK_FALLBACK[benchmark]


def _institution_cell(marks: tuple[InstitutionMark, ...]) -> str:
    if not marks:
        return "--"
    parts = []
    for mark in marks:
        by_benchmark: dict[str, list[str]] = {}
        for benchmark, label in mark.centres:
            by_benchmark.setdefault(benchmark, []).append(_CENTRE_NAMES.get(label, label))
        where = ", ".join(f"{_benchmark_name(b)} {'/'.join(ls)}" for b, ls in by_benchmark.items())
        parts.append(rf"{INSTITUTIONS[mark.institution].name} ({where})~\cite{{{mark.source}}}")
    return "; ".join(parts)


def _selection_cell(selection: Selection | None) -> str:
    return "--" if selection is None else rf"{selection.label}~\cite{{{selection.source}}}"


def _matrix_row(matrix: ExposureMatrix, model: str) -> str:
    cells = [model]
    cells += [_CELL[matrix.status(model, d.domain)] for d in SOURCE_DATASETS]
    cells += [_institution_cell(matrix.institutions(model)), _selection_cell(matrix.selection(model))]
    return " & ".join(cells) + r" \\"


def _count_row(panel: str, counts: dict[str, Count]) -> str:
    cells = [rf"\textit{{{panel.capitalize()}: exposed / undisclosed}}"]
    cells += [f"{counts[d.domain].exposed} / {counts[d.domain].undisclosed}" for d in SOURCE_DATASETS]
    return " & ".join(cells + ["", ""]) + r" \\"


_MATRIX_CAPTION = (
    r"\caption{\textbf{Pretraining exposure of every encoder to the source datasets the "
    r"benchmarks score.} Columns: TCGA (scored by TCGA-2x2, TCGA-4x4 and the Tolkach-ESCA TCGA "
    r"cohort held out of its main score), CAMELYON16/17 (Camelyon), PANDA (PCaBiop in domain), "
    r"PAR (PCaBiop out of domain) and Tolkach-ESCA (its UKK, WNS and CHA cohorts). A cell is "
    r"\textbf{exposed} when the encoder's disclosed pretraining corpus includes the dataset, "
    r"\emph{not exposed} when the disclosed corpus excludes it, and \emph{undisclosed} when its "
    r"authors have not disclosed the corpus; no cell is inferred. Every cell is resolved from "
    r"\code{model\_metadata.csv} by the resolver that places the exposure daggers of the Pareto "
    r"figures and results tables. \emph{Shared institution} marks a disclosed corpus drawn from "
    r"the institution behind a scored centre (benchmark and centre in parentheses): possible "
    r"source-domain overlap only, kept apart from dataset exposure; it does not establish that any "
    r"scored slide was seen. \emph{PathoROB in development} is the benchmark its authors "
    r"disclose using while developing the encoder, a different kind of exposure; -- means none "
    r"disclosed. Counts exclude the natural-image control. Overlap is flagged here, not adjusted "
    r"for.}"
)


def render_matrix(matrix: ExposureMatrix, fingerprints: list[tuple[str, str]]) -> str:
    """``tab:exposure-matrix``: tile rows, slide rows, the control, then the counts."""
    counts = matrix.counts()
    heads = ["Encoder", *(DATASET_HEADS[d.domain] for d in SOURCE_DATASETS)]
    heads += [r"\shortstack{Shared\\institution}", r"\shortstack{PathoROB in\\development}"]
    n_cols = len(heads)
    lines = [HEADER, *fingerprint_lines(fingerprints)]
    lines += [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}l{'c' * len(SOURCE_DATASETS)}p{{3.6cm}}p{{1.7cm}}@{{}}}}",
        r"\hline",
        " & ".join(heads) + r" \\",
        r"\hline",
    ]
    for panel in ("tile", "slide"):
        models = [row.model for row in matrix.rows if row.panel == panel]
        if not models:
            continue
        lines.append(rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{_PANEL_HEADS[panel]}}}}} \\")
        lines += [_matrix_row(matrix, model) for model in models]
        lines.append(r"\hline")
    controls = [row.model for row in matrix.rows if row.panel == "control"]
    if controls:
        lines += [_matrix_row(matrix, model) for model in controls]
        lines.append(r"\hline")
    lines += [_count_row(panel, counts[panel]) for panel in ("tile", "slide")]
    lines += [
        r"\hline",
        r"\end{tabular}",
        _MATRIX_CAPTION,
        r"\label{tab:exposure-matrix}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


#: Columns a sensitivity frame carries: one row per encoder, CRoMa already a margin.
SCORE_COLUMNS = ("croma", "croma_ltm_alpha", "nipd_id", "nipd_ood")
#: The correlation settings the caption states; passed explicitly so the two cannot drift.
_INFERENCE = {"n_boot": 2000, "level": 0.95, "seed": 0, "n_permutations": 10_000}


@dataclass(frozen=True)
class Variant:
    """One panel of encoders and what the headline associations and frontier are on it."""

    name: str
    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    #: The kept encoders' scores, indexed by model: the full panel's rows, never recomputed.
    scores: pd.DataFrame
    rho_id: CorrelationInference
    rho_ood: CorrelationInference
    #: Median-vs-LTM_10 frontier, in the staircase order the figure draws.
    frontier: tuple[str, ...]
    top: tuple[str, ...]

    @property
    def unchanged(self) -> bool:
        """Nothing was dropped, so this variant is the full panel."""
        return not self.dropped


def _variant(name: str, scores: pd.DataFrame, kept: set[str]) -> Variant:
    from _panel_macros import median_tail_frontier, top_by_median

    if len(kept) < 3:
        raise CaptionClaimError(
            f"{name}: {len(kept)} encoders kept; a correlation and a top three need at least three."
        )
    frame = scores.loc[sorted(kept)]
    ranked = frame.rename_axis("model").reset_index()
    return Variant(
        name=name,
        kept=tuple(frame.index),
        dropped=tuple(sorted(set(scores.index) - kept)),
        scores=frame,
        rho_id=spearman_inference(frame["croma"], frame["nipd_id"], **_INFERENCE),
        rho_ood=spearman_inference(frame["croma"], frame["nipd_ood"], **_INFERENCE),
        frontier=tuple(median_tail_frontier(ranked)),
        top=tuple(top_by_median(ranked)["model"]),
    )


def sensitivity(
    scores: pd.DataFrame, *, drop: set[str], keep: set[str]
) -> tuple[Variant, Variant, Variant]:
    """The full ranked panel, the panel without ``drop``, and the panel of ``keep`` only.

    ``scores`` has a ``model`` column and ``SCORE_COLUMNS``; the control is a floor and is
    removed here. Each variant subsets the full panel's rows, so no remaining encoder's score
    moves: only the correlations, the frontier and the top three are recomputed.
    """
    frame = scores[scores["model"] != CONTROL_MODEL].set_index("model")[list(SCORE_COLUMNS)]
    panel = set(frame.index)
    for label, names in (("drop", drop), ("keep", keep)):
        unknown = sorted(set(names) - panel)
        if unknown:
            raise CaptionClaimError(f"{label} set names encoders outside the panel: {unknown}")
    return (
        _variant("full", frame, panel),
        _variant("drop-exposed", frame, panel - set(drop)),
        _variant("disclosed-unexposed", frame, set(keep)),
    )


def rank_inputs(scores: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median CRoMa (margin) and LTM_10, encoder x benchmark, over the ranked panel.

    ``scores`` maps a benchmark to its ``load_scores`` frame, so the mean-rank rows read the
    same runs, under the same root, as the per-benchmark rows. The per-run columns are aligned
    into one frame as ``_cross_benchmark.load`` aligns them, so the ``method="first"`` tie rule
    sees the rows in the order the overview figure's ranks do. Every benchmark must rank the
    same panel: a mean rank over encoders some benchmark never ranked is not a mean.
    """
    columns = {
        label: frame[frame["model"] != CONTROL_MODEL].set_index("model")
        for label, frame in scores.items()
    }
    (first, panel), *rest = ((label, set(f.index)) for label, f in columns.items())
    for label, other in rest:
        if other != panel:
            raise CaptionClaimError(
                f"{label} does not rank {first}'s panel; only in {label} {sorted(other - panel)}, "
                f"only in {first} {sorted(panel - other)}. The mean-rank rows need one panel."
            )
    medians = pd.DataFrame({label: f["croma"] for label, f in columns.items()})
    tails = pd.DataFrame({label: f["croma_ltm_alpha"] for label, f in columns.items()})
    return medians, tails.reindex(medians.index)


def rank_frontier(medians: pd.DataFrame, tails: pd.DataFrame, kept: set[str]) -> tuple[str, ...]:
    """The cross-benchmark mean-rank frontier over ``kept`` only, every rank re-taken among them.

    ``medians`` and ``tails`` are ``rank_inputs``. Ranks are relative to who competes, so a
    dropped encoder must leave the ranking, not just the plot: both ranks are taken among the
    kept encoders (1 = highest median, 1 = mildest tail), with the tie rule ``_rank_pareto``
    uses. The frontier is ``RankPareto.frontier``, the primitive the overview figure rings;
    its exposure and adversarial fields play no part in it.
    """
    from _rank_pareto import RankPareto

    unknown = sorted(set(kept) - set(medians.index))
    if unknown:
        raise CaptionClaimError(f"rank frontier: {unknown} are not in the ranked panel.")
    order = [m for m in medians.index if m in kept]
    sub = RankPareto(
        medians=medians.loc[order],
        median_ranks=medians.loc[order].rank(ascending=False, method="first").astype(int),
        tail_ranks=tails.loc[order].rank(ascending=False, method="first").astype(int),
        exposed=frozenset(),
        adversarial="",
    )
    return sub.frontier


@dataclass(frozen=True)
class BenchmarkSensitivity:
    """One benchmark's three variants, under its printed name."""

    name: str
    variants: tuple[Variant, Variant, Variant]


@dataclass(frozen=True)
class RankVariant:
    """One variant of the cross-benchmark mean-rank frontier."""

    name: str
    kept: tuple[str, ...]
    frontier: tuple[str, ...]


_VARIANT_LABELS = {
    "full": "All ranked",
    "drop-exposed": "Drop exposed",
    "disclosed-unexposed": "Disclosed, not exposed",
}
_UNCHANGED = "none exposed to the scored dataset; identical to all ranked"
_SENSITIVITY_COLUMNS = 7


def p_text(p: float) -> str:
    """A permutation p as printed: three decimals, bounded below at 0.001, never ``p=0.000``."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _rho_cell(inference: CorrelationInference) -> str:
    import math

    lo, hi = inference.ci.lo, inference.ci.hi
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise CaptionClaimError(
            f"rho over n={inference.n} has no bootstrap interval; every resample was degenerate."
        )
    return (
        rf"${bare_num(inference.rho)}$ $[{bare_num(lo)}, {bare_num(hi)}]$, "
        rf"${p_text(inference.p_value)}$"
    )


def _sensitivity_rows(block: BenchmarkSensitivity) -> list[str]:
    rows = []
    for i, variant in enumerate(block.variants):
        cells = [block.name if i == 0 else "", _VARIANT_LABELS[variant.name], str(len(variant.kept))]
        if variant.name == "drop-exposed" and variant.unchanged:
            cells.append(rf"\multicolumn{{{_SENSITIVITY_COLUMNS - 3}}}{{@{{}}l}}{{{_UNCHANGED}}}")
        else:
            cells += [
                _rho_cell(variant.rho_id),
                _rho_cell(variant.rho_ood),
                ", ".join(variant.frontier),
                ", ".join(variant.top),
            ]
        rows.append(" & ".join(cells) + r" \\")
    return rows


def _rank_rows(ranks: tuple[RankVariant, ...]) -> list[str]:
    return [
        " & ".join(
            ["Mean rank" if i == 0 else "", _VARIANT_LABELS[v.name], str(len(v.kept)), "--", "--",
             ", ".join(v.frontier), "--"]
        )
        + r" \\"
        for i, v in enumerate(ranks)
    ]


def _sensitivity_caption(blocks: list[BenchmarkSensitivity]) -> str:
    from _paper_tables import CROMA_HEADLINE_M

    unchanged = [b.name for b in blocks if b.variants[1].unchanged]
    parts = [
        r"\caption{\textbf{Headline associations and frontiers without the exposed encoders.}",
        r"Per benchmark, over the ranked pathology encoders (control excluded): \emph{All ranked}",
        r"is the full panel; \emph{Drop exposed} removes the encoders whose disclosed corpus",
        r"includes the benchmark's scored dataset (Table~\ref{tab:exposure-matrix});",
        r"\emph{Disclosed, not exposed} also removes the encoders whose corpus is undisclosed.",
        r"$\rho_{\mathrm{ID}}$ and $\rho_{\mathrm{OOD}}$ are Spearman correlations of",
        rf"median $\mcode{{CRoMa}}(m{{=}}{CROMA_HEADLINE_M})$ with nIPD$_{{\mathrm{{ID}}}}$ and",
        rf"nIPD$_{{\mathrm{{OOD}}}}$, with a {round(_INFERENCE['level'] * 100)}\% encoder-bootstrap",
        rf"percentile interval ({_INFERENCE['n_boot']:,} resamples) and a two-sided permutation",
        rf"$p$, exact over all $n!$ orderings for $n \le {EXACT_PERMUTATION_MAX_N}$ and from",
        rf"{_INFERENCE['n_permutations']:,} seeded permutations otherwise. The frontier is the",
        r"median--$\mcode{LTM}_{10}$",
        r"Pareto frontier the Pareto figures ring, in staircase order; the top three are by median",
        r"\code{CRoMa}. The mean-rank rows re-rank the encoders kept on all three benchmarks and give",
        r"the cross-benchmark mean-rank frontier: \emph{Drop exposed} removes the encoders exposed on",
        r"any of the three, \emph{Disclosed, not exposed} keeps those not exposed on any. Dropping",
        r"encoders does not change any remaining encoder's score; only the correlations, ranks and",
        r"frontiers are recomputed.",
    ]
    if unchanged:
        names = " and ".join(unchanged)
        parts.append(
            rf"No encoder's disclosed corpus includes the dataset {names} scores, so its"
            r" \emph{Drop exposed} row is the full panel; shared-institution markers"
            r" (Table~\ref{tab:exposure-matrix}) are flagged, never dropped."
        )
    parts.append(r"Exposure is flagged, not adjusted for.}")
    return " ".join(parts)


def render_drop_exposed(
    blocks: list[BenchmarkSensitivity],
    ranks: tuple[RankVariant, ...],
    fingerprints: list[tuple[str, str]],
) -> str:
    """``tab:drop-exposed``: three variants per benchmark, then the mean-rank frontier."""
    lines = [HEADER, *fingerprint_lines(fingerprints)]
    lines += [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrccp{3.0cm}p{2.6cm}@{}}",
        r"\hline",
        r"Benchmark & Encoders & $n$ & $\rho_{\mathrm{ID}}$ [95\% CI], $p$ & "
        r"$\rho_{\mathrm{OOD}}$ [95\% CI], $p$ & Frontier & Top three \\",
        r"\hline",
    ]
    for block in blocks:
        lines += _sensitivity_rows(block)
        lines.append(r"\hline")
    lines += _rank_rows(ranks)
    lines += [
        r"\hline",
        r"\end{tabular}",
        _sensitivity_caption(blocks),
        r"\label{tab:drop-exposed}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


#: Macro stems of the dataset columns and the non-full variants.
_DATASET_STEMS = {
    "tcga": "Tcga",
    "camelyon": "Camelyon",
    "panda": "Panda",
    "par": "Par",
    "tolkach-esca": "TolkachEsca",
}
_VARIANT_STEMS = {"drop-exposed": "DropExposed", "disclosed-unexposed": "DisclosedUnexposed"}


def _newcommand(name: str, body: str) -> str:
    return rf"\newcommand{{\{name}}}{{{body}}}"


def exposure_macros(
    matrix: ExposureMatrix,
    blocks: dict[str, BenchmarkSensitivity],
    ranks: tuple[RankVariant, ...],
) -> list[str]:
    """Numbers only: per-dataset counts, and per benchmark (keyed by manifest prefix) and
    variant the panel size, both correlations and the frontier; then the mean-rank frontier."""
    from _paper_tables import num_math

    lines = ["% Pretraining exposure counts (tab:exposure-matrix)"]
    for panel, counts in matrix.counts().items():
        for domain, count in counts.items():
            stem = f"Exposure{panel.capitalize()}{_DATASET_STEMS[domain]}"
            lines.append(_newcommand(f"{stem}NExposed", str(count.exposed)))
            lines.append(_newcommand(f"{stem}NUndisclosed", str(count.undisclosed)))
    lines.append("% Drop-exposed sensitivity (tab:drop-exposed)")
    for prefix, block in blocks.items():
        for variant in block.variants[1:]:
            stem = prefix + _VARIANT_STEMS[variant.name]
            lines.append(_newcommand(f"{stem}N", str(len(variant.kept))))
            for regime, inference in (("Id", variant.rho_id), ("Ood", variant.rho_ood)):
                lines.append(_newcommand(f"{stem}Nipd{regime}Rho", num_math(inference.rho)))
                lines.append(
                    _newcommand(
                        f"{stem}Nipd{regime}Ci",
                        rf"$[{bare_num(inference.ci.lo)}, {bare_num(inference.ci.hi)}]$",
                    )
                )
                lines.append(_newcommand(f"{stem}Nipd{regime}P", rf"${p_text(inference.p_value)}$"))
            lines.append(_newcommand(f"{stem}ParetoFrontierModels", ", ".join(variant.frontier)))
            lines.append(_newcommand(f"{stem}ParetoFrontierNModels", str(len(variant.frontier))))
    for variant in ranks:
        if variant.name == "full":
            continue
        stem = "TileRankFrontier" + _VARIANT_STEMS[variant.name]
        lines.append(_newcommand(f"{stem}Models", ", ".join(sorted(variant.frontier))))
        lines.append(_newcommand(f"{stem}NModels", str(len(variant.frontier))))
        lines.append(_newcommand(f"{stem}NKept", str(len(variant.kept))))
    return lines


def _run_input(root: Path, entry) -> Path:
    """``entry``'s ``metrics.csv`` under ``root``; a missing run fails with what to re-run."""
    path = Path(root) / entry.metrics_rel
    if not path.is_file():
        raise FileNotFoundError(f"{entry.prefix}: missing {path}; re-run the benchmark.")
    return path


def _join_input(root: Path) -> Path:
    """The APD join under ``root``; a missing join fails with what to re-run."""
    from _panel_macros import APD_JOINED_CSV

    path = Path(root) / APD_JOINED_CSV
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run scripts/studies/apd/apd_croma_correlation.py.")
    return path


def load_scores(root: Path, entry) -> pd.DataFrame:
    """One row per encoder of ``entry``'s run: median CRoMa (margin) and LTM_10 from the run's
    ``metrics.csv``, nIPD_ID and nIPD_OOD from the APD join, in published spellings.

    The run is resolved through ``paper_manifest`` (``entry.metrics_rel``), never a spelled-out
    protocol. The join carries its own copy of each encoder's CRoMa; if that is not the run's,
    the join was built from another run and its nIPD cannot be paired with these scores.
    """
    import numpy as np
    from _paper_tables import croma_as_margin
    from loaders import DATASETS

    metrics_path, joined_path = _run_input(root, entry), _join_input(root)
    (dataset,) = [k for k, spec in DATASETS.items() if spec["benchmark"] == entry.benchmark]
    metrics = published_models(pd.read_csv(metrics_path)).set_index("model")
    joined = published_models(pd.read_csv(joined_path))
    joined = joined[joined["dataset"] == dataset].set_index("model")

    if set(metrics.index) != set(joined.index):
        raise CaptionClaimError(
            f"{entry.prefix}: the run and the APD join disagree on the panel; only in the run "
            f"{sorted(set(metrics.index) - set(joined.index))}, only in the join "
            f"{sorted(set(joined.index) - set(metrics.index))}."
        )
    # The join is on the margin scale whatever the run stores (``loaders.load_joined``
    # normalises), so the run is compared, and reported, as margins too.
    run_croma = croma_as_margin(metrics["croma"])
    join_croma = joined["croma"].astype(float).reindex(run_croma.index)
    stale = run_croma.index[~np.isclose(run_croma, join_croma, rtol=0.0, atol=1e-12)]
    if len(stale):
        raise CaptionClaimError(
            f"{entry.prefix}: the APD join's CRoMa is not the run's for {sorted(stale)}; "
            "re-run scripts/studies/apd/apd_croma_correlation.py against this run."
        )
    out = pd.DataFrame(
        {
            "croma": run_croma,
            "croma_ltm_alpha": metrics["croma_ltm_alpha"].astype(float),
            "nipd_id": joined["nipd_id"].astype(float).reindex(run_croma.index),
            "nipd_ood": joined["nipd_ood"].astype(float).reindex(run_croma.index),
        }
    )
    return out.rename_axis("model").reset_index()


# --- the build ------------------------------------------------------------------------------

#: The benchmarks of ``tab:drop-exposed`` and of the mean-rank frontier, by manifest prefix.
SENSITIVITY_PREFIXES = ("Camelyon", "TcgaFourByFour", "Tolkach")
MATRIX_OUT = Path("paper/sections/supp/exposure_matrix.tex")
DROP_EXPOSED_OUT = Path("paper/sections/supp/drop_exposed.tex")


def fingerprints(base: Path, rels: list[str]) -> list[tuple[str, str]]:
    """``(path, sha256)`` for each input, path as given (relative to ``base``)."""
    import hashlib

    return [(rel, hashlib.sha256((Path(base) / rel).read_bytes()).hexdigest()) for rel in rels]


def check_fingerprints(base: Path, recorded: list[tuple[str, str]]) -> None:
    """Refuse a table whose inputs no longer hash to what it was built from.

    ``output/`` is shared, so a run or study can rewrite an input while this build reads it; a
    table recording one fingerprint and showing numbers from another would misstate its source.
    """
    now = dict(fingerprints(base, [rel for rel, _ in recorded]))
    changed = sorted(rel for rel, digest in recorded if now[rel] != digest)
    if changed:
        raise CaptionClaimError(
            f"input(s) changed while the exposure tables were built: {changed}; rebuild."
        )


@dataclass(frozen=True)
class Built:
    """Both tables and the macros, from one read of the inputs."""

    matrix_tex: str
    drop_exposed_tex: str
    macros: list[str]


def _rank_variants(scores: dict[str, pd.DataFrame], matrix: ExposureMatrix):
    """The mean-rank frontier on the full panel, without encoders exposed on any of the three
    benchmarks, and on the encoders disclosed and unexposed on all three. ``scores`` is keyed
    by manifest prefix."""
    from paper_manifest import by_prefix

    medians, tails = rank_inputs(scores)
    panel = set(medians.index)
    benchmarks = [by_prefix(p).benchmark for p in scores]
    exposed_any = set().union(*(matrix.drop_exposed(b, panel) for b in benchmarks))
    clean_all = set.intersection(*(matrix.disclosed_unexposed(b, panel) for b in benchmarks))
    order = tuple(sorted(panel))
    variants = []
    for name, kept in (
        ("full", panel),
        ("drop-exposed", panel - exposed_any),
        ("disclosed-unexposed", clean_all),
    ):
        if len(kept) < 2:
            raise CaptionClaimError(f"mean-rank {name}: {len(kept)} encoders kept, no frontier.")
        variants.append(
            RankVariant(
                name, tuple(m for m in order if m in kept), rank_frontier(medians, tails, kept)
            )
        )
    return tuple(variants)


def build(root: Path = REPO) -> Built:
    """Read every input once, under ``root``, refuse on a missing or moving one, and render.

    The mean-rank variants are ranked from the same scores as the per-benchmark rows, so the
    whole table, and the fingerprints in its header, describe one set of runs.
    """
    from _panel_macros import APD_JOINED_CSV
    from paper_manifest import by_prefix

    entries = [by_prefix(p) for p in SENSITIVITY_PREFIXES]
    # Hash before reading: a fingerprint taken after the read could describe a replacement
    # written in between, and the closing check would then pass over the old scores.
    for entry in entries:
        _run_input(root, entry)
    _join_input(root)
    md_prints = fingerprints(REPO, [str(METADATA.relative_to(REPO))])
    run_prints = fingerprints(root, [e.metrics_rel for e in entries] + [APD_JOINED_CSV])

    scores = {e.prefix: load_scores(root, e) for e in entries}
    matrix = exposure_matrix(load_metadata())
    blocks = {}
    for entry in entries:
        roster = set(scores[entry.prefix]["model"]) - {CONTROL_MODEL}
        blocks[entry.prefix] = BenchmarkSensitivity(
            entry.short_name,
            sensitivity(
                scores[entry.prefix],
                drop=matrix.drop_exposed(entry.benchmark, roster),
                keep=matrix.disclosed_unexposed(entry.benchmark, roster),
            ),
        )
    ranks = _rank_variants(scores, matrix)

    built = Built(
        matrix_tex=render_matrix(matrix, md_prints),
        drop_exposed_tex=render_drop_exposed(list(blocks.values()), ranks, md_prints + run_prints),
        macros=exposure_macros(matrix, blocks, ranks),
    )
    check_fingerprints(REPO, md_prints)
    check_fingerprints(root, run_prints)
    return built


def load_metadata(path: Path = METADATA) -> pd.DataFrame:
    """model_metadata.csv with literal cells kept and published model names.

    ``keep_default_na=False`` is load-bearing: pandas otherwise reads the control's literal
    ``n/a`` as NaN. The names are restyled because every surface built from the matrix is a
    published one, and the daggers resolve against published rosters too.
    """
    frame = pd.read_csv(path, keep_default_na=False, na_values=[])
    return published_models(frame, columns=("model", "parent_model"))
