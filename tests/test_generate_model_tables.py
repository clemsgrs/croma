"""Builder tests for the model-table generator (``scripts/repro``).

The pure builders (``summary_table_body``, ``_wsis_tiles_cell``) are exercised on a
small synthetic metadata frame, plus a couple of fidelity checks against the
committed ``model_metadata.csv``. No ``paper/`` tree is required. Prior art:
``tests/test_analyze_results.py``.
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

# Columns the summary builder reads; the synthetic rows below fill the ones each
# builder touches and default the rest to empty.
COLUMNS = [
    "model",
    "panel",
    "panel_order",
    "method",
    "params",
    "n_wsis",
    "n_tiles",
    "wsis_source",
    "dim",
    "corpus",
]


def _row(**kw: object) -> dict[str, object]:
    base = {col: "" for col in COLUMNS}
    base.update(kw)
    return base


def _synthetic_df() -> pd.DataFrame:
    """A minimal 2-panel metadata frame covering the summary render branches.

    - ``TcgaModel`` / ``CardModel`` : fully rendered tile rows.
    - ``MarkedModel`` : #WSIs sourced from PathoROB (superscript marker); tiles n/d.
    - ``SlideModel``  : slide panel, both counts undisclosed (n/d collapse).
    """
    rows = [
        _row(
            model="TcgaModel",
            panel="tile",
            panel_order=1,
            method="iBOT",
            params=r"$86$M",
            n_wsis=r"$6$k",
            n_tiles=r"$43$M",
            wsis_source="card",
            dim=r"$768$",
            corpus=r"\textbf{TCGA (public)}",
        ),
        _row(
            model="CardModel",
            panel="tile",
            panel_order=2,
            method="DINOv2",
            params=r"$307$M",
            n_wsis=r"$100$k",
            n_tiles=r"$100$M",
            wsis_source="card",
            dim=r"$1024$",
            corpus="Vendor (prop.)",
        ),
        _row(
            model="MarkedModel",
            panel="tile",
            panel_order=3,
            method="vision--language",
            params="n/d",
            n_wsis=r"$350$k",
            n_tiles="n/d",
            wsis_source="pathorob",
            dim=r"$768$",
            corpus="PMC-OA",
        ),
        _row(
            model="SlideModel",
            panel="slide",
            panel_order=1,
            method="grid ViT",
            params="n/d",
            n_wsis="n/d",
            n_tiles="n/d",
            wsis_source="card",
            dim=r"$768$",
            corpus="Vendor (prop.)",
        ),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_summary_body_renders_both_panels_and_rows() -> None:
    body = gmt.summary_table_body(_synthetic_df())

    assert r"\multicolumn{6}{l}{\emph{Tile-level encoders}} \\" in body
    assert r"\multicolumn{6}{l}{\emph{Slide-level encoders}} \\" in body
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


# --------------------------------------------------------------------------
# Fidelity checks against the committed single-source metadata.
# --------------------------------------------------------------------------
def test_committed_metadata_sources_every_wsi_count_from_a_model_card() -> None:
    """No row is PathoROB-sourced any more, so the rendered summary carries no dagger.

    CONCH, CONCHv1.5, UNI2-h and Prost40M used to take their #WSIs from PathoROB and mark it
    with a superscript dagger. Their counts now come from the model cards, which also disclose
    parameter and tile counts PathoROB does not, so the marker has no users left. The mechanism
    itself stays -- a future row may need it -- and is covered against a synthetic row by
    test_summary_body_marks_pathorob_sourced_wsis above.
    """
    df = gmt.load_metadata()
    assert set(df.loc[df["wsis_source"] == "pathorob", "model"]) == set()

    body = gmt.summary_table_body(df)
    assert gmt.PATHOROB_WSIS_MARKER not in body
    assert r"UNI2-h~\cite{uni2h} & DINOv2 & $681$M & ${>}350$k / ${>}200$M" in body
    assert r"CONCHv1.5~\cite{titan} & vision--language & $307$M & $100$k / $100$M" in body
    assert r"CONCH~\cite{conch} & vision--language & $86$M & $21.4$k / $16$M" in body
    assert r"Prost40M~\cite{grisi2026bcr} & DINO & $22$M & $2$k / ${\sim}40$M" in body


def test_waiv_finetunes_inherit_parent_pretraining_provenance() -> None:
    """Waiv variants add acquisition robustness without rewriting their pretraining history."""
    metadata = gmt.load_metadata().set_index("model")
    inherited = [
        "params",
        "n_wsis",
        "n_tiles",
        "wsis_source",
        "regime",
        "corpus",
        "tcga_exposed",
        "disclosed",
        "disclosed_sources",
        "source_marker",
        "corpus_domains",
    ]

    for model, parent in [("Mascaret", "Midnight-12k"), ("Phaet", "Phikon-v2")]:
        row = metadata.loc[model]
        assert row["parent_model"] == parent
        assert row["variant_role"] == "robustness-finetune"
        assert row["adaptation"] == "acquisition-robustness"
        assert row["family"] == "waiv"
        assert row["cite"] == "filiot2026robustifying"
        assert row[inherited].to_dict() == metadata.loc[parent, inherited].to_dict()

    body = gmt.summary_table_body(metadata.reset_index())
    assert (
        r"Mascaret~\cite{filiot2026robustifying} & acquisition-robustness fine-tune of \code{Midnight-12k}"
        in body
    )
    assert (
        r"Phaet~\cite{filiot2026robustifying} & acquisition-robustness fine-tune of \code{Phikon-v2}"
        in body
    )


def test_rudolfv2_metadata_describes_one_teacher_student_training_run() -> None:
    metadata = gmt.load_metadata().set_index("model")
    names = ["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S"]
    family = metadata.loc[names]

    assert family["family"].tolist() == ["rudolfv2"] * 3
    assert family["variant_role"].tolist() == ["teacher", "distilled-student", "distilled-student"]
    assert family["parent_model"].tolist() == ["", "RudolfV 2", "RudolfV 2"]
    assert family["training_run"].nunique() == 1
    assert family["shared_corpus"].nunique() == 1
    assert family["params"].tolist() == [r"$1.1$B", r"$86$M", r"$22$M"]
    assert family["dim"].tolist() == [r"$3072$", r"$1536$", r"$768$"]
    assert family["n_wsis"].tolist() == [r"${>}300$k"] * 3
    assert family["n_tiles"].tolist() == ["n/d"] * 3
    assert family["disclosed"].tolist() == ["yes"] * 3
    assert family["tcga_exposed"].tolist() == [False] * 3
    assert family["cite"].tolist() == ["rudolfv2"] * 3

    body = gmt.summary_table_body(metadata.reset_index())
    assert r"RudolfV 2~\cite{rudolfv2} & DINOv2 (teacher)" in body
    assert r"RudolfV 2-B~\cite{rudolfv2} & distilled from \code{RudolfV 2}" in body
    assert r"RudolfV 2-S~\cite{rudolfv2} & distilled from \code{RudolfV 2}" in body
    assert body.count(r"Charit\'e + LMU (shared family corpus)") == 3


#: Methods that denote a vision--language model however the cell names them. The ``method``
#: column records the architecture where the source paper does (TITAN is a CoCa, i.e. a
#: contrastive captioner; the PRISM family is a Perceiver aggregator trained against
#: clinical report text), so the regime cannot be read off a single literal.
VL_METHODS = {"vision--language", "CoCa", "Perceiver"}


def test_regime_rule_matches_method() -> None:
    df = gmt.load_metadata()
    for _, row in df.iterrows():
        expected = "VLFM" if row["method"] in VL_METHODS else "vision-only"
        assert row["regime"] == expected, row["model"]


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
