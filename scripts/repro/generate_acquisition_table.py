r"""Render the per-centre acquisition details (``supp/acquisition.tex``, ``tab:acquisition``).

One row per benchmark x centre: scanner, tile size, resolution, magnification, and the
slides, patients and tiles of each role the centre plays. ``_acquisition`` joins the
curated facts in ``acquisition.csv`` to the counted cohorts; this script only lays them
out. A curated "not documented" prints as n.d.; a column that does not apply (a slide-level
benchmark's tiles and tile size) as a dash.

The caption's TCGA sentence is phrased from the barcodes and quotes their counts through
the ``\<prefix>Barcode*`` macros ``generate_paper_values.py`` emits from the same function.

Usage:
  python scripts/repro/generate_acquisition_table.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _acquisition as acq
from _acquisition import REPO

OUT = REPO / "paper/sections/supp/acquisition.tex"
LABEL = "tab:acquisition"

ND = "n.d."
NA = "--"


def _n(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _tex(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _fact(value: str) -> str:
    if value == acq.NOT_DOCUMENTED:
        return ND
    value = _tex(value).replace("~", r"$\sim$")
    return value[:-1] + r"$\times$" if value.endswith("x") and value[:-1][-1:].isdigit() else value


def _letters(rows: list[acq.Row]) -> dict[str, str]:
    """One footnote letter per distinct note, in order of first use."""
    letters: dict[str, str] = {}
    for row in rows:
        for note in filter(None, row.note.split("|")):
            letters.setdefault(note, "abcdefghijklmnopqrstuvwxyz"[len(letters)])
    return letters


def _centre(row: acq.Row, letters: dict[str, str]) -> str:
    marks = [letters[n] for n in filter(None, row.note.split("|"))]
    return _tex(row.centre) + (rf"$^{{{','.join(marks)}}}$" if marks else "")


def _line(row: acq.Row, first: bool, letters: dict[str, str]) -> str:
    cells = [
        _tex(row.benchmark) if first else "",
        _centre(row, letters),
        _fact(row.scanner),
        NA if row.tile_px is None else str(row.tile_px),
        _fact(row.mpp),
        _fact(row.magnification),
        _n(row.slides),
        ND if row.patients is None else _n(row.patients),
        NA if row.tiles is None else _n(row.tiles),
        ", ".join(row.roles),
    ]
    return " & ".join(cells) + r" \\"


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _tcga_sentence(prefix: str, name: str, b: acq.Barcodes) -> str:
    unexpected = sorted(set(b.sample_slides) - {acq.PRIMARY, acq.RECURRENT, acq.METASTATIC})
    if unexpected:
        raise ValueError(f"{name}: no sentence for TCGA sample type(s) {unexpected}")
    m = lambda key: rf"$\{prefix}{key}$"  # noqa: E731
    dx = (
        f"all {m('BarcodeSlides')} slides are"
        if b.diagnostic == b.slides
        else f"{m('DxSlides')} of {m('BarcodeSlides')} slides are"
    )
    patients = "one per patient" if b.patients == b.slides else f"from {m('BarcodePatients')} patients"
    others = []
    recurrent = b.sample_slides.get(acq.RECURRENT, 0)
    metastatic = b.sample_slides.get(acq.METASTATIC, 0)
    if recurrent:
        others.append(
            f"{m('RecurrentSlides')} recurrent-tumour (02) {_plural(recurrent, 'slide')} of "
            f"{m('RecurrentTiles')} tiles"
        )
    if metastatic:
        others.append(f"{m('MetastaticSlides')} metastatic (06) {_plural(metastatic, 'slide')}")
    types = "all are primary tumour (sample type 01)"
    if others:
        types += " except " + " and ".join(others)
    tail = "" if metastatic else "; none is metastatic (06)"
    return f"On {name}, {dx} diagnostic (DX) slides, {patients}, and {types}{tail}."


def _caption(rows: list[acq.Row], barcodes: dict[str, acq.Barcodes]) -> str:
    cites = list(dict.fromkeys(c for row in rows for c in row.cite))
    tcga = " ".join(
        _tcga_sentence(prefix, acq.display_name(prefix), b) for prefix, b in barcodes.items()
    )
    return (
        r"\textbf{Acquisition details per benchmark and centre.} Scanner, resolution (mpp, "
        r"microns per pixel) and magnification are those stated by the publications that "
        rf"release each dataset~\cite{{{','.join(cites)}}}; {ND}, not documented there. The "
        r"resolution is the tiles' on the tile-level benchmarks and the scan's on PCaBiop, "
        r"whose slide encoders tile each slide at their own resolution, so its tile size "
        rf"and tile count do not apply ({NA}). Tile size is read from the tile images, one per "
        r"slide. Slides, patients and tiles count the rows each role evaluates: the "
        r"representation cohort is scored by \code{RI}, \code{MaRI} and \code{CRoMa}; the "
        r"in-domain (ID) probe pool trains and tests the confounder-biased probes, and the "
        r"out-of-domain (OOD) probe pool is only tested on (Supplementary "
        r"Table~\ref{tab:split-schedule}). A centre whose roles use the same slides and "
        r"tiles takes one row. Patients are counted where slide names identify them, from "
        r"TCGA barcodes and PAR file names. " + tcga
    )


def build(rows: list[acq.Row], barcodes: dict[str, acq.Barcodes], label: str = LABEL) -> str:
    letters = _letters(rows)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llp{2.8cm}rllrrrp{2.2cm}}",
        r"\hline",
        r"Benchmark & Centre & Scanner & Tile (px) & mpp & Mag. & Slides & Patients & Tiles "
        r"& Role \\",
        r"\hline",
    ]
    previous = None
    for row in rows:
        if previous is not None and row.benchmark != previous:
            lines.append(r"\hline")
        lines.append(_line(row, row.benchmark != previous, letters))
        previous = row.benchmark
    lines += [r"\hline", r"\end{tabular}"]
    if letters:
        notes = " ".join(rf"$^{{{letter}}}${_tex(note)}" for note, letter in letters.items())
        lines += ["", r"\vspace{3pt}", rf"\parbox{{\linewidth}}{{\scriptsize {notes}}}"]
    lines += [
        rf"\caption{{{_caption(rows, barcodes)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    tex = build(acq.rows(acq.load_curated(), acq.load_cohorts()), acq.load_barcodes())
    if args.check:
        if not out.exists() or out.read_text() != tex:
            print(f"STALE  {shown}", file=sys.stderr)
            return 1
        print(f"ok     {shown}", file=sys.stderr)
        return 0
    if not out.parent.exists():
        print("skip   paper/ absent", file=sys.stderr)
        return 0
    out.write_text(tex)
    print(f"wrote  {shown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
