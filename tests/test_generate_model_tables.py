"""Builder tests for the model-table generator (``scripts/repro``).

The pure builders (``summary_table_body``, ``provenance_grid``,
``provenance_table_body``) are exercised on a small synthetic metadata frame, plus
a couple of fidelity checks against the committed ``model_metadata.csv``. No
``paper/`` tree is required. Prior art: ``tests/test_analyze_results.py``.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

import generate_model_tables as gmt  # noqa: E402

# Full column set the builders read; the synthetic rows below fill the ones each
# builder touches and default the rest to empty.
COLUMNS = [
    "model", "panel", "panel_order", "method", "params", "n_wsis", "n_tiles",
    "wsis_source", "dim", "regime", "corpus", "tcga_exposed", "disclosed",
    "disclosed_sources", "source_marker", "corpus_domains", "prov_group",
    "prov_order", "prov_cell_override", "prov_cell_markers",
]


def _row(**kw: object) -> dict[str, object]:
    base = {col: "" for col in COLUMNS}
    base.update(kw)
    return base


def _synthetic_df() -> pd.DataFrame:
    """A minimal 2-panel metadata frame covering the derivation branches.

    - ``TcgaModel``   : TCGA-exposed, disclosed (drives ID/OOD derivation).
    - ``CardModel``   : non-exposed disclosed model (all-OOD baseline).
    - ``MarkedModel`` : #WSIs sourced from PathoROB (superscript marker).
    - ``SlideModel``  : undisclosed, slide panel (all n/d + grouping).
    """
    rows = [
        _row(
            model="TcgaModel", panel="tile", panel_order=1, method="iBOT",
            params=r"$86$M", n_wsis=r"$6$k", n_tiles=r"$43$M", wsis_source="card",
            dim=r"$768$", regime="vision-only", corpus=r"\textbf{TCGA (public)}",
            tcga_exposed=True, disclosed="yes", disclosed_sources="TCGA only",
            corpus_domains="tcga", prov_group="TcgaModel", prov_order=1,
        ),
        _row(
            model="CardModel", panel="tile", panel_order=2, method="DINOv2",
            params=r"$307$M", n_wsis=r"$100$k", n_tiles=r"$100$M",
            wsis_source="card", dim=r"$1024$", regime="vision-only",
            corpus="Vendor (prop.)", tcga_exposed=False, disclosed="yes",
            disclosed_sources="none", corpus_domains="", prov_group="CardModel",
            prov_order=2,
        ),
        _row(
            model="MarkedModel", panel="tile", panel_order=3,
            method="vision--language", params="n/d", n_wsis=r"$350$k",
            n_tiles="n/d", wsis_source="pathorob", dim=r"$768$", regime="VLFM",
            corpus="PMC-OA", tcga_exposed=False, disclosed="no",
            disclosed_sources="Institute X", corpus_domains="",
            prov_group="MarkedModel", prov_order=4,
        ),
        _row(
            model="SlideModel", panel="slide", panel_order=1, method="grid ViT",
            params="n/d", n_wsis="n/d", n_tiles="n/d", wsis_source="card",
            dim=r"$768$", regime="vision-only", corpus="Vendor (prop.)",
            tcga_exposed=False, disclosed="no", disclosed_sources="Institute Y",
            corpus_domains="", prov_group="SlideModel", prov_order=3,
        ),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_summary_body_renders_both_panels_and_rows() -> None:
    body = gmt.summary_table_body(_synthetic_df())

    assert r"\multicolumn{6}{l}{\emph{Tile-level encoders} (3-model PathoROB panel)} \\" in body
    assert r"\multicolumn{6}{l}{\emph{Slide-level encoders} (PANDA panel)} \\" in body
    # A fully rendered, non-marked row.
    assert r"CardModel & DINOv2 & $307$M & $100$k / $100$M & $1024$ & Vendor (prop.) \\" in body
    # Both counts undisclosed collapse to a single "n/d".
    assert r"SlideModel & grid ViT & n/d & n/d & $768$ & Vendor (prop.) \\" in body


def test_summary_body_marks_pathorob_sourced_wsis() -> None:
    body = gmt.summary_table_body(_synthetic_df())

    assert gmt.PATHOROB_WSIS_MARKER == r"$^{\dagger}$"
    # The PathoROB-sourced #WSIs carries the marker; the tiles side stays n/d.
    assert r"MarkedModel & vision--language & n/d & $350$k$^{\dagger}$ / n/d & $768$" in body
    # The card-sourced model's #WSIs must NOT carry the marker.
    card_row = r"CardModel & DINOv2 & $307$M & $100$k / $100$M & $1024$ & Vendor (prop.) \\"
    assert card_row in body
    assert gmt.PATHOROB_WSIS_MARKER not in card_row


def test_provenance_grid_derives_id_ood_from_corpus_domain() -> None:
    grid = gmt.provenance_grid(_synthetic_df()).set_index("model")

    # A TCGA-exposed model: ID (bold) on the TCGA benchmark, OOD on Camelyon.
    assert grid.loc["TcgaModel", "TCGA"] == r"\textbf{ID}"
    assert grid.loc["TcgaModel", "Camelyon"] == "OOD"
    # Tolkach ID is partial (TCGA sub-cohort) and carries the star marker.
    assert grid.loc["TcgaModel", "Tolkach"] == r"\textbf{ID}$^{*}$"
    assert grid.loc["TcgaModel", "PANDA"] == "OOD"

    # A disclosed model with no benchmark-domain overlap is OOD everywhere.
    for benchmark in gmt.BENCHMARKS:
        assert grid.loc["CardModel", benchmark] == "OOD"

    # Undisclosed composition -> n/d across every benchmark.
    for benchmark in gmt.BENCHMARKS:
        assert grid.loc["SlideModel", benchmark] == "n/d"


def test_provenance_grid_splits_disclosed_and_undisclosed_sections() -> None:
    grid = gmt.provenance_grid(_synthetic_df()).set_index("model")

    assert grid.loc["TcgaModel", "section"] == "disclosed"
    assert grid.loc["CardModel", "section"] == "disclosed"
    assert grid.loc["SlideModel", "section"] == "undisclosed"

    body = gmt.provenance_table_body(_synthetic_df())
    assert r"\multicolumn{6}{l}{\emph{Pretraining composition disclosed}} \\" in body
    assert (
        r"\multicolumn{6}{l}{\emph{Pretraining composition not disclosed "
        r"(proprietary corpora)}} \\" in body
    )


# --------------------------------------------------------------------------
# Fidelity checks against the committed single-source metadata.
# --------------------------------------------------------------------------
def test_committed_metadata_marks_the_four_pathorob_wsis() -> None:
    df = gmt.load_metadata()
    marked = set(df.loc[df["wsis_source"] == "pathorob", "model"])
    assert marked == {"CONCH", "CONCHv1.5", "UNI2-h", "Prost40M"}

    body = gmt.summary_table_body(df)
    assert r"UNI2-h & DINOv2 & n/d & $350$k$^{\dagger}$ / n/d" in body
    assert r"CONCHv1.5 & vision--language & n/d & $100$k$^{\dagger}$ / n/d" in body
    assert r"CONCH & vision--language & $86$M & $21.4$k$^{\dagger}$ / $16$M" in body
    assert r"Prost40M & DINO & ${\sim}22$M & $2$k$^{\dagger}$ / ${\sim}40$M" in body


def test_committed_metadata_reproduces_key_provenance_cells() -> None:
    grid = gmt.provenance_grid(gmt.load_metadata()).set_index("model")

    # TCGA-exposed tile model: ID on TCGA, OOD on Camelyon (the headline rule).
    assert grid.loc["Midnight-12k", "TCGA"] == r"\textbf{ID}"
    assert grid.loc["Midnight-12k", "Camelyon"] == "OOD"
    # Special-case markers preserved from the ground-truth table.
    assert grid.loc["Prost40M", "Camelyon"] == r"ID$^{\S}$"
    assert grid.loc["Prost40M", "PANDA"] == "ID"
    assert grid.loc["MOOZY", "PANDA"] == r"\textbf{ID}$^{\P}$"
    assert grid.loc["CONCHv1.5", "sources"].endswith(r"$^{\ddagger}$")
    # Grouped undisclosed row collapses the two models onto one line.
    assert grid.loc["UNI, UNI2-h", "TCGA"] == "n/d"


def test_regime_rule_matches_method() -> None:
    df = gmt.load_metadata()
    for _, row in df.iterrows():
        expected = "VLFM" if row["method"] == "vision--language" else "vision-only"
        assert row["regime"] == expected


def test_render_template_requires_placeholder() -> None:
    with pytest.raises(ValueError):
        gmt.render_template("no placeholder here", "body")
    assert gmt.render_template("a\n%%BODY%%\nb", "X") == "a\nX\nb"


def test_not_applicable_cells_survive_the_csv_round_trip() -> None:
    """``n/a`` must reach the table, not be eaten by pandas' default NaN parsing.

    The natural-image control has no #WSIs -- that is *not applicable*, a different
    claim from *undisclosed*. pandas maps the literal string ``n/a`` to NaN unless
    told otherwise, which would silently render an empty cell.
    """
    df = gmt.load_metadata()
    control = df[df["model"] == "DINOv2-B"].iloc[0]
    assert control["n_wsis"] == "n/a"
    assert control["n_tiles"] == "n/a"


def test_wsis_tiles_cell_collapses_a_uniform_missing_token() -> None:
    assert gmt._wsis_tiles_cell("n/a", "n/a", "card") == "n/a"
    assert gmt._wsis_tiles_cell("n/d", "n/d", "card") == "n/d"
    # A half-known cell keeps both halves rather than collapsing.
    assert gmt._wsis_tiles_cell("$350$k", "n/d", "card") == "$350$k / n/d"


def test_gpfm_is_in_distribution_on_every_benchmark() -> None:
    """GPFM pretrains on TCGA, CAMELYON16/17 and PANDA, so no benchmark is held out.

    It is the only tile model for which this holds; the direct-inclusion marker (\\P)
    must appear on the two benchmarks whose slides are literally in its corpus.
    """
    grid = gmt.provenance_grid(gmt.load_metadata()).set_index("model")
    assert grid.loc["GPFM", "Camelyon"] == r"\textbf{ID}$^{\P}$"
    assert grid.loc["GPFM", "PANDA"] == r"\textbf{ID}$^{\P}$"
    assert grid.loc["GPFM", "TCGA"] == r"\textbf{ID}"
    assert grid.loc["GPFM", "Tolkach"] == r"\textbf{ID}$^{*}$"


def test_natural_image_control_is_out_of_distribution_everywhere() -> None:
    grid = gmt.provenance_grid(gmt.load_metadata()).set_index("model")
    assert all(grid.loc["DINOv2-B", b] == "OOD" for b in gmt.BENCHMARKS)
