"""Generate the model-summary LaTeX table BODY from the single committed metadata
source ``scripts/repro/model_metadata.csv``.

Only the ``\\hline``-separated row body is generated. The float wrapper, header
row, ``\\caption`` and prose footnotes live in the hand-maintained template
``templates/model_summary.tex.tmpl``, which has a single ``%%BODY%%`` line the
generator replaces. The assembled ``.tex`` is written into ``paper/sections/``
only when that (git-ignored, local-only) tree exists; the body builder is a pure
function of a metadata ``DataFrame`` and is callable/testable without the
``paper/`` tree.

Metadata columns read here (single source of truth):
  model, panel, panel_order, method, params, n_wsis, n_tiles, wsis_source, dim,
  corpus, cite

Run: python scripts/repro/generate_model_tables.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "model_metadata.csv"
TEMPLATE_DIR = HERE / "templates"
SUMMARY_TMPL = TEMPLATE_DIR / "model_summary.tex.tmpl"
PAPER_SECTIONS = HERE.parents[1] / "paper" / "sections"
BODY_PLACEHOLDER = "%%BODY%%"

# Superscript that flags a #WSIs count taken from PathoROB rather than the model
# card. Its footnote prose lives in the summary template; the symbol must agree.
PATHOROB_WSIS_MARKER = r"$^{\dagger}$"


def _s(value: object) -> str:
    """CSV cell as a stripped string; NaN/empty -> ``""``."""
    if value is None:
        return ""
    text = str(value)
    return "" if text in ("nan", "NaN", "") else text


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
def _panel_separator(panel: str) -> str:
    if panel == "tile":
        return r"\multicolumn{6}{l}{\emph{Tile-level encoders}} \\"
    return r"\multicolumn{6}{l}{\emph{Slide-level encoders}} \\"


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


def _cite(row: pd.Series) -> str:
    """The ``~\\cite{key}`` appended to a model name, or ``""`` when no key is recorded.

    The key comes from the metadata's ``cite`` column (a bib key in ``paper/references.bib``),
    so the model table cites each encoder without a second hand-maintained citation list.
    """
    key = _s(row.get("cite", "")) if "cite" in row.index else ""
    return rf"~\cite{{{key}}}" if key else ""


def _summary_row(row: pd.Series) -> str:
    cells = [
        _s(row["model"]) + _cite(row),
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
        block = [_panel_separator(panel), r"\hline"]
        block += [_summary_row(row) for _, row in sub.iterrows()]
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


def main() -> None:
    df = load_metadata()
    tex = build_summary_tex(df)
    out = PAPER_SECTIONS / "model_summary.tex"
    if PAPER_SECTIONS.exists():
        out.write_text(tex)
        print(f"wrote {out}")
    else:
        # paper/ is a local-only, git-ignored tree; nothing to write in a fresh
        # checkout. Report the generated body size so the run is still useful.
        rows = tex.count(chr(92) + chr(92))
        print(f"model_summary.tex: {rows} rows (\\\\); paper/ absent, not written")


if __name__ == "__main__":
    main()
