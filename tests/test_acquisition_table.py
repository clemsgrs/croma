r"""The acquisition-details table (``tab:acquisition``) and the TCGA barcode macros.

One row per benchmark x centre states the scanner, tile size, resolution, slides, patients,
tiles and the role the centre plays. Counts come from the benchmark views and the APD
metadata; scanner, resolution and magnification are curated in ``acquisition.csv``, where a
value is either sourced or written "not documented". The expected values below are worked
out by hand from small synthetic cohorts and from the TCGA barcode specification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts/repro", ROOT / "scripts/bench", ROOT / "scripts/studies/apd"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _acquisition as acq  # noqa: E402
import generate_acquisition_table as table  # noqa: E402

HEADER = "benchmark,centre,name,scanner,mpp,magnification,patient_id,note,cite,source\n"


def _curated(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "acquisition.csv"
    path.write_text(HEADER + "".join(r + "\n" for r in rows))
    return path


class TestCuratedFacts:
    def test_the_shipped_table_loads(self):
        curated = acq.load_curated()
        assert {"pathorob-camelyon", "pathorob-tcga-4x4", "panda"} <= set(curated["benchmark"])

    def test_a_documented_value_without_a_source_raises(self, tmp_path):
        path = _curated(tmp_path, "b,X,X,Leica GT 450,0.25,40x,,,,")
        with pytest.raises(ValueError, match="no source"):
            acq.load_curated(path)

    def test_an_undocumented_row_needs_no_source(self, tmp_path):
        path = _curated(tmp_path, "b,X,X,not documented,not documented,not documented,,,,")
        assert len(acq.load_curated(path)) == 1

    def test_a_blank_value_raises_instead_of_reading_as_unknown(self, tmp_path):
        path = _curated(tmp_path, "b,X,X,,0.5,20x,,,pathorob,PathoROB")
        with pytest.raises(ValueError, match="not documented"):
            acq.load_curated(path)

    def test_a_centre_listed_twice_raises(self, tmp_path):
        row = "b,X,X,not documented,not documented,not documented,,,,"
        with pytest.raises(ValueError, match="twice"):
            acq.load_curated(_curated(tmp_path, row, row))


class TestTcgaBarcodes:
    #: Barcode fields: project-TSS-participant-sample/vial-portion/analyte-slide.
    SLIDES = {
        "TCGA-E9-A1R5-01Z-00-DX1": 30,  # primary tumour, diagnostic
        "TCGA-50-5066-02Z-00-DX1": 30,  # recurrent tumour
        "TCGA-AA-0001-06Z-00-DX1": 30,  # metastatic
        "TCGA-AA-0001-01Z-00-TS1": 10,  # same patient, frozen tissue slide
    }

    def test_counts_patients_sample_types_and_diagnostic_slides(self):
        b = acq.tcga_barcodes(self.SLIDES)
        assert (b.slides, b.patients, b.diagnostic) == (4, 3, 3)
        assert b.sample_slides == {"01": 2, "02": 1, "06": 1}
        assert b.sample_tiles == {"01": 40, "02": 30, "06": 30}

    def test_a_slide_name_that_is_not_a_barcode_raises(self):
        with pytest.raises(ValueError, match="not a TCGA slide barcode"):
            acq.tcga_barcodes({"test_035": 300})

    def test_macros_state_the_counts_the_prose_cites(self):
        lines = acq.barcode_macros("TcgaFourByFour", acq.tcga_barcodes(self.SLIDES))
        assert lines == [
            r"\newcommand{\TcgaFourByFourBarcodeSlides}{4}",
            r"\newcommand{\TcgaFourByFourBarcodePatients}{3}",
            r"\newcommand{\TcgaFourByFourDxSlides}{3}",
            r"\newcommand{\TcgaFourByFourPrimarySlides}{2}",
            r"\newcommand{\TcgaFourByFourRecurrentSlides}{1}",
            r"\newcommand{\TcgaFourByFourRecurrentTiles}{30}",
            r"\newcommand{\TcgaFourByFourMetastaticSlides}{1}",
        ]

    def test_thousands_are_grouped_for_latex(self):
        slides = {f"TCGA-AA-{i:04d}-01Z-00-DX1": 30 for i in range(1975)}
        lines = acq.barcode_macros("TcgaTwoByTwo", acq.tcga_barcodes(slides))
        assert r"\newcommand{\TcgaTwoByTwoBarcodePatients}{1{,}975}" in lines
        assert r"\newcommand{\TcgaTwoByTwoRecurrentTiles}{0}" in lines


def _png(path: Path, px: int) -> str:
    from PIL import Image

    Image.new("RGB", (px, px)).save(path)
    return str(path)


def _cohort(tmp_path: Path, *, probe=True, px: dict[str, int] | None = None) -> acq.Cohort:
    """Centre A: the same 2 slides x 3 tiles represented and probed in domain. Centre B:
    2 slides represented, 3 probed in domain (one more). Centre C: 1 slide of 2 tiles held
    out. Slide c005a/c005b are one PAR-style patient's two slides."""
    slides = {"A": ["c001", "c002"], "B": ["c003", "c004"], "C": ["c005a"]}
    view = pd.DataFrame(
        [(f"{s}__{t}", c, s) for c in "AB" for s in slides[c] for t in range(3)],
        columns=["sample_id", "confounder", "group_id"],
    )
    md = [("ID", s, t, "A") for s in slides["A"] for t in range(3)]
    md += [("ID", s, t, "B") for s in [*slides["B"], "c005b"] for t in range(3)]
    md += [("OOD", "c005a", t, "C") for t in range(2)]
    metadata = pd.DataFrame(md, columns=["subset", "slide_id", "patch_id", "medical_center"])
    px = px or {}
    tileset = pd.DataFrame(
        [
            (_png(tmp_path / f"{s}.png", px.get(c, 8)), c, s)
            for c, ss in {"A": slides["A"], "B": [*slides["B"], "c005b"], "C": ["c005a"]}.items()
            for s in ss
        ],
        columns=["image_path", "confounder", "group_id"],
    )
    return acq.Cohort(
        benchmark="b",
        name="Bench",
        representation=view,
        probe=metadata if probe else None,
        tileset=tileset,
    )


def _facts(tmp_path: Path, centres="ABC", patient_id="par-filename") -> pd.DataFrame:
    rows = [
        f"b,{c},Centre {c},Scanner {c},0.5,20x,{patient_id},,pathorob,PathoROB" for c in centres
    ]
    return acq.load_curated(_curated(tmp_path, *rows))


class TestRows:
    def test_one_row_per_centre_and_role_merging_identical_pools(self, tmp_path):
        rows = acq.rows(_facts(tmp_path), [_cohort(tmp_path)])
        got = [(r.centre, r.roles, r.slides, r.patients, r.tiles, r.tile_px) for r in rows]
        assert got == [
            ("Centre A", ("Representation", "ID probe"), 2, 2, 6, 8),
            ("Centre B", ("Representation",), 2, 2, 6, 8),
            ("Centre B", ("ID probe",), 3, 3, 9, 8),
            ("Centre C", ("OOD probe",), 1, 1, 2, 8),
        ]
        assert (rows[0].benchmark, rows[0].scanner, rows[0].mpp) == ("Bench", "Scanner A", "0.5")

    def test_patients_without_an_identifier_rule_are_not_counted(self, tmp_path):
        rows = acq.rows(_facts(tmp_path, patient_id=""), [_cohort(tmp_path)])
        assert {r.patients for r in rows} == {None}

    def test_a_tile_size_that_varies_within_a_centre_raises(self, tmp_path):
        cohort = _cohort(tmp_path)
        _png(Path(cohort.tileset["image_path"].iloc[0]), 16)
        with pytest.raises(ValueError, match="tile sizes"):
            acq.rows(_facts(tmp_path), [cohort])

    def test_a_centre_with_no_curated_row_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"no curated row for \['C'\]"):
            acq.rows(_facts(tmp_path, centres="AB"), [_cohort(tmp_path)])

    def test_a_curated_centre_the_runs_never_evaluate_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"evaluate no rows for \['D'\]"):
            acq.rows(_facts(tmp_path, centres="ABCD"), [_cohort(tmp_path)])

    def test_a_pooled_row_counts_the_whole_cohort_once(self, tmp_path):
        facts = acq.load_curated(
            _curated(tmp_path, "b,*,{n} centres,not documented,0.5,20x,,,pathorob,PathoROB")
        )
        cohort = _cohort(tmp_path, probe=False)
        (row,) = acq.rows(facts, [cohort])
        assert (row.centre, row.roles, row.slides, row.tiles) == (
            "2 centres",
            ("Representation",),
            4,
            12,
        )

    def test_a_slide_level_cohort_has_no_tiles_or_tile_size(self, tmp_path):
        cohort = _cohort(tmp_path)
        slide_level = acq.Cohort("b", "Bench", cohort.representation, cohort.probe, None)
        rows = acq.rows(_facts(tmp_path), [slide_level])
        assert {(r.tiles, r.tile_px) for r in rows} == {(None, None)}


def _row(**changes) -> acq.Row:
    base = dict(
        benchmark="Camelyon",
        centre="RUMC",
        scanner="3DHistech Pannoramic Flash II 250",
        tile_px=256,
        mpp="0.5",
        magnification="20x",
        slides=34,
        patients=None,
        tiles=10200,
        roles=("Representation", "ID probe"),
        note="",
        cite=("pathorob",),
    )
    return acq.Row(**{**base, **changes})


def _barcodes(**changes) -> dict[str, acq.Barcodes]:
    base = dict(
        slides=272,
        patients=272,
        diagnostic=272,
        sample_slides={"01": 271, "02": 1},
        sample_tiles={"01": 8130, "02": 30},
    )
    return {"TcgaFourByFour": acq.Barcodes(**{**base, **changes})}


def _body(tex: str) -> list[str]:
    return [line for line in tex.splitlines() if line.endswith(r"\\") and " & " in line]


class TestTable:
    def test_a_row_states_every_column(self):
        (line,) = _body(table.build([_row()], _barcodes()))[1:]
        assert line == (
            r"Camelyon & RUMC & 3DHistech Pannoramic Flash II 250 & 256 & 0.5 & 20$\times$ & "
            r"34 & n.d. & 10{,}200 & Representation, ID probe \\"
        )

    def test_undocumented_and_inapplicable_cells_are_told_apart(self):
        row = _row(scanner=acq.NOT_DOCUMENTED, mpp="~0.75", tile_px=None, tiles=None)
        (line,) = _body(table.build([row], _barcodes()))[1:]
        assert r" & n.d. & -- & $\sim$0.75 & " in line
        assert line.split(" & ")[8] == "--"

    def test_the_benchmark_is_named_once_per_block(self):
        rows = [_row(), _row(centre="UMCU"), _row(benchmark="TCGA-4x4", centre="Asterand")]
        names = [line.split(" & ")[0] for line in _body(table.build(rows, _barcodes()))[1:]]
        assert names == ["Camelyon", "", "TCGA-4x4"]

    def test_notes_become_shared_footnotes_in_order_of_first_use(self):
        rows = [
            _row(note="Scanned at RUMC."),
            _row(centre="RST", note="Artificial patients.|Scanned at RUMC."),
        ]
        tex = table.build(rows, _barcodes())
        centres = [line.split(" & ")[1] for line in _body(tex)[1:]]
        assert centres == ["RUMC$^{a}$", "RST$^{b,a}$"]
        assert r"$^{a}$Scanned at RUMC. $^{b}$Artificial patients." in tex

    def test_the_caption_cites_every_source_once(self):
        rows = [_row(cite=("litjens2018camelyon", "pathorob")), _row(cite=("pathorob",))]
        assert r"\cite{litjens2018camelyon,pathorob}" in table.build(rows, _barcodes())

    def test_the_tcga_sentence_states_the_barcode_provenance(self):
        tex = table.build([_row()], _barcodes())
        assert (
            r"On TCGA-4x4, all $\TcgaFourByFourBarcodeSlides$ slides are diagnostic (DX) "
            r"slides, one per patient, and all are primary tumour (sample type 01) except "
            r"$\TcgaFourByFourRecurrentSlides$ recurrent-tumour (02) slide of "
            r"$\TcgaFourByFourRecurrentTiles$ tiles; none is metastatic (06)."
        ) in tex

    def test_the_tcga_sentence_follows_the_data_when_it_changes(self):
        tex = table.build(
            [_row()],
            _barcodes(
                patients=270,
                diagnostic=271,
                sample_slides={"01": 270, "06": 2},
                sample_tiles={"01": 8100, "06": 60},
            ),
        )
        assert (
            r"On TCGA-4x4, $\TcgaFourByFourDxSlides$ of $\TcgaFourByFourBarcodeSlides$ slides "
            r"are diagnostic (DX) slides, from $\TcgaFourByFourBarcodePatients$ patients, and "
            r"all are primary tumour (sample type 01) except "
            r"$\TcgaFourByFourMetastaticSlides$ metastatic (06) slides."
        ) in tex


def _embedded(benchmark: str) -> bool:
    import benchmarks
    import layout

    return layout.tileset_manifest(benchmarks.get(benchmark).tileset).is_file()


@pytest.mark.skipif(
    not (_embedded("pathorob-tcga-4x4") and _embedded("pathorob-tcga-2x2")),
    reason="the TCGA tilesets are not embedded in this checkout",
)
def test_the_live_tcga_slides_are_diagnostic_primary_and_one_per_patient():
    """The provenance the caption states: every embedded TCGA slide is a diagnostic slide
    from its own patient, all primary tumour except one recurrent-tumour slide of 30 tiles,
    and none metastatic."""
    b = acq.load_barcodes()
    four, two = b["TcgaFourByFour"], b["TcgaTwoByTwo"]
    assert (four.slides, four.patients, four.diagnostic) == (272, 272, 272)
    assert four.sample_slides == {"01": 271, "02": 1}
    assert four.sample_tiles["02"] == 30
    assert (two.slides, two.patients, two.diagnostic) == (1975, 1975, 1975)
    assert two.sample_slides == {"01": 1975}
