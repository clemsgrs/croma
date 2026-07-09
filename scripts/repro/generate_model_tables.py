"""Generate the model-summary and model-provenance LaTeX table BODIES from the
single committed metadata source ``scripts/repro/model_metadata.csv``.

Only the ``\\hline``-separated row bodies are generated. The float wrapper, header
row, ``\\caption`` and prose footnotes live in the hand-maintained templates
``templates/model_summary.tex.tmpl`` and ``templates/model_provenance.tex.tmpl``;
each has a single ``%%BODY%%`` line the generator replaces. The assembled ``.tex``
is written into ``paper/sections/`` only when that (git-ignored, local-only) tree
exists; the body builders are pure functions of a metadata ``DataFrame`` and are
callable/testable without the ``paper/`` tree.

Metadata columns (single source of truth):
  model, panel, panel_order, method, params, n_wsis, n_tiles, wsis_source, dim,
  regime, corpus, tcga_exposed, disclosed, disclosed_sources, source_marker,
  corpus_domains, prov_group, prov_order, prov_cell_override, prov_cell_markers

Run: python scripts/repro/generate_model_tables.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "model_metadata.csv"
TEMPLATE_DIR = HERE / "templates"
SUMMARY_TMPL = TEMPLATE_DIR / "model_summary.tex.tmpl"
PROVENANCE_TMPL = TEMPLATE_DIR / "model_provenance.tex.tmpl"
PAPER_SECTIONS = HERE.parents[1] / "paper" / "sections"
BODY_PLACEHOLDER = "%%BODY%%"

# Superscript that flags a #WSIs count taken from PathoROB rather than the model
# card. Its footnote prose lives in the summary template; the symbol must agree.
PATHOROB_WSIS_MARKER = r"$^{\dagger}$"

# ---------------------------------------------------------------------------
# Provenance derivation: benchmark <-> pretraining-domain overlap.
# ---------------------------------------------------------------------------
BENCHMARKS = ["Camelyon", "TCGA", "Tolkach", "PANDA"]
# Each benchmark's pretraining-overlap domain token. Tolkach-ESCA mixes one TCGA
# sub-cohort with three non-TCGA cohorts, so its ID is via the ``tcga`` token and
# is flagged partial (the ``*`` marker) below.
DOMAIN_OF = {"Camelyon": "camelyon", "TCGA": "tcga", "Tolkach": "tcga", "PANDA": "panda"}
# Symbolic marker name -> LaTeX symbol (rendered as a math superscript).
MARKERS = {"star": r"*", "sec": r"\S", "par": r"\P", "ddag": r"\ddagger", "dag": r"\dagger"}


def _s(value: object) -> str:
    """CSV cell as a stripped string; NaN/empty -> ``""``."""
    if value is None:
        return ""
    text = str(value)
    return "" if text in ("nan", "NaN", "") else text


def _mark(name: str) -> str:
    """Render a symbolic marker name as a LaTeX math superscript, e.g. ``$^{\\S}$``."""
    return rf"$^{{{MARKERS[name]}}}$"


def _domains(spec: str) -> set[str]:
    """Parse the ``corpus_domains`` cell (``;``-separated tokens) into a set."""
    return {tok.strip() for tok in _s(spec).split(";") if tok.strip()}


def load_metadata(path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load the model metadata CSV with empty cells normalised to ``""``.

    ``keep_default_na=False`` is load-bearing: pandas otherwise parses the literal
    cell ``n/a`` (not applicable -- the natural-image control has no #WSIs) as NaN,
    which ``fillna("")`` would then render as an empty table cell.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[])


# ---------------------------------------------------------------------------
# Model-summary table body.
# ---------------------------------------------------------------------------
def _panel_separator(panel: str, n_rows: int) -> str:
    if panel == "tile":
        return (
            rf"\multicolumn{{6}}{{l}}{{\emph{{Tile-level encoders}} "
            rf"({n_rows}-model PathoROB panel)}} \\"
        )
    return r"\multicolumn{6}{l}{\emph{Slide-level encoders} (PANDA panel)} \\"


def _wsis_tiles_cell(n_wsis: str, n_tiles: str, wsis_source: str) -> str:
    """Render the ``WSIs / tiles`` cell.

    When both counts are undisclosed the original tables collapse the cell to a
    single ``n/d``. A ``#WSIs`` value sourced from PathoROB carries a superscript
    marker whose footnote (in the template) cites PathoROB.
    """
    n_wsis, n_tiles = _s(n_wsis), _s(n_tiles)
    for token in ("n/d", "n/a"):
        if n_wsis == token and n_tiles == token:
            return token
    if wsis_source == "pathorob":
        n_wsis = n_wsis + PATHOROB_WSIS_MARKER
    return f"{n_wsis} / {n_tiles}"


def _summary_row(row: pd.Series) -> str:
    cells = [
        _s(row["model"]),
        _s(row["method"]),
        _s(row["params"]),
        _wsis_tiles_cell(row["n_wsis"], row["n_tiles"], _s(row["wsis_source"])),
        _s(row["dim"]),
        _s(row["corpus"]),
    ]
    return " & ".join(cells) + r" \\"


def summary_table_body(df: pd.DataFrame) -> str:
    """Build the model-summary table body (panel separators + model rows).

    Pure function of the metadata frame; emits every panel present, ordered by
    ``panel_order``, with an interior ``\\hline`` between panels.
    """
    blocks: list[list[str]] = []
    for panel in ["tile", "slide"]:
        sub = df[df["panel"] == panel].sort_values("panel_order")
        if sub.empty:
            continue
        block = [_panel_separator(panel, len(sub)), r"\hline"]
        block += [_summary_row(row) for _, row in sub.iterrows()]
        blocks.append(block)

    lines: list[str] = []
    for i, block in enumerate(blocks):
        if i > 0:
            lines.append(r"\hline")
        lines.extend(block)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model-provenance grid + table body.
# ---------------------------------------------------------------------------
def _derive_cell(benchmark: str, domains: set[str]) -> str:
    """ID/OOD for one (benchmark, disclosed-corpus) pair from domain overlap.

    A model is in-distribution on a benchmark iff the benchmark's domain token is
    in the model's disclosed corpus. Tolkach ID is always partial (only its TCGA
    sub-cohort overlaps) and carries the ``*`` marker.
    """
    if DOMAIN_OF[benchmark] in domains:
        cell = r"\textbf{ID}"
        if benchmark == "Tolkach":
            cell += _mark("star")
        return cell
    return "OOD"


def _parse_override(spec: str) -> dict[str, str]:
    """Parse a full 4-cell override, e.g. ``ID:sec|ID:sec|ID:sec|ID:``.

    Token grammar per cell: ``VALUE[:MARKER]`` where VALUE is ``ID`` (plain),
    ``BID`` (bold ID), ``OOD`` or ``ND`` (n/d) and MARKER is a name in ``MARKERS``.
    Cells map positionally onto ``BENCHMARKS``.
    """
    cells: dict[str, str] = {}
    for benchmark, token in zip(BENCHMARKS, spec.split("|")):
        value, _, marker = token.strip().partition(":")
        value = value.strip()
        if value == "BID":
            cell = r"\textbf{ID}"
        elif value == "ND":
            cell = "n/d"
        else:
            cell = value  # "ID" or "OOD"
        if marker.strip():
            cell += _mark(marker.strip())
        cells[benchmark] = cell
    return cells


def _parse_cell_markers(spec: str) -> dict[str, str]:
    """Parse ``Benchmark:marker`` pairs (``;``-separated) appended to derived cells."""
    out: dict[str, str] = {}
    for token in _s(spec).split(";"):
        token = token.strip()
        if not token:
            continue
        benchmark, _, marker = token.partition(":")
        out[benchmark.strip()] = marker.strip()
    return out


def provenance_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the per-model ID/OOD/n-d provenance grid from the metadata.

    One row per ``prov_group`` (grouped undisclosed models collapse to a single
    row; a model appearing in both summary panels collapses too), ordered by
    ``prov_order``. Columns: ``model, sources, section`` + one per benchmark.
    """
    ordered = df.sort_values(["prov_order", "panel", "panel_order"])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, meta in ordered.iterrows():
        group = _s(meta["prov_group"])
        if group in seen:
            continue
        seen.add(group)

        disclosed = _s(meta["disclosed"]).lower() == "yes"
        sources = _s(meta["disclosed_sources"])
        source_marker = _s(meta["source_marker"])
        if source_marker:
            sources += _mark(source_marker)

        if not disclosed:
            cells = {benchmark: "n/d" for benchmark in BENCHMARKS}
        else:
            override = _s(meta["prov_cell_override"])
            if override:
                cells = _parse_override(override)
            else:
                domains = _domains(meta["corpus_domains"])
                cells = {b: _derive_cell(b, domains) for b in BENCHMARKS}
                for benchmark, marker in _parse_cell_markers(
                    meta["prov_cell_markers"]
                ).items():
                    cells[benchmark] += _mark(marker)

        rows.append(
            {
                "model": group,
                "sources": sources,
                "section": "disclosed" if disclosed else "undisclosed",
                **cells,
            }
        )
    return pd.DataFrame(rows, columns=["model", "sources", "section", *BENCHMARKS])


def _provenance_row(row: pd.Series) -> str:
    cells = [row["model"], row["sources"], *[row[b] for b in BENCHMARKS]]
    return " & ".join(cells) + r" \\"


def provenance_table_body(df: pd.DataFrame) -> str:
    """Build the provenance table body (section separators + derived grid rows)."""
    grid = provenance_grid(df)
    sections = [
        ("disclosed", r"\multicolumn{6}{l}{\emph{Pretraining composition disclosed}} \\"),
        (
            "undisclosed",
            r"\multicolumn{6}{l}{\emph{Pretraining composition not disclosed "
            r"(proprietary corpora)}} \\",
        ),
    ]
    blocks: list[list[str]] = []
    for name, separator in sections:
        sub = grid[grid["section"] == name]
        if sub.empty:
            continue
        block = [separator, r"\hline"]
        block += [_provenance_row(row) for _, row in sub.iterrows()]
        blocks.append(block)

    lines: list[str] = []
    for i, block in enumerate(blocks):
        if i > 0:
            lines.append(r"\hline")
        lines.extend(block)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template assembly.
# ---------------------------------------------------------------------------
def render_template(template_text: str, body: str) -> str:
    """Replace the single ``%%BODY%%`` line in a template with the generated body."""
    if BODY_PLACEHOLDER not in template_text:
        raise ValueError(f"template is missing the {BODY_PLACEHOLDER!r} placeholder")
    return template_text.replace(BODY_PLACEHOLDER, body)


def build_summary_tex(df: pd.DataFrame, template_path: Path = SUMMARY_TMPL) -> str:
    return render_template(template_path.read_text(), summary_table_body(df))


def build_provenance_tex(df: pd.DataFrame, template_path: Path = PROVENANCE_TMPL) -> str:
    return render_template(template_path.read_text(), provenance_table_body(df))


def main() -> None:
    df = load_metadata()
    outputs = {
        "model_summary.tex": build_summary_tex(df),
        "model_provenance.tex": build_provenance_tex(df),
    }
    if PAPER_SECTIONS.exists():
        for name, tex in outputs.items():
            (PAPER_SECTIONS / name).write_text(tex)
            print(f"wrote {PAPER_SECTIONS / name}")
    else:
        # paper/ is a local-only, git-ignored tree; nothing to write in a fresh
        # checkout. Report the generated body sizes so the run is still useful.
        for name, tex in outputs.items():
            print(f"{name}: {tex.count(chr(92) + chr(92))} rows (\\\\); paper/ absent, not written")


if __name__ == "__main__":
    main()
