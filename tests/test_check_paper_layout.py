"""``scripts/repro/check_paper_layout.py``: the local gate against floats that do not fit.

CI has no TeX, so nothing there can see a table run off the page. This check reads the logs
``paper/build.sh`` leaves behind and names every ``Overfull \\hbox`` over 1 pt and every
``Float too large``, with the source file that was open when pdflatex reported it.

The fixtures are real pdflatex logs: ``main_excerpt.log`` is a stretch of the manuscript's
own log on the 27-encoder panel (TinyTeX, which does not wrap log lines), and
``wrapped_at_79.log`` is a small document compiled with TeX Live's default 79-character
log width, so file names are split across lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

import check_paper_layout as cpl  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "paper_layout"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="latin-1")


def _summary(findings: list[cpl.Finding]) -> list[tuple[str, float, str]]:
    return [(f.kind, round(f.points, 2), f.source) for f in findings]


def test_attributes_each_problem_to_the_file_open_when_it_was_reported() -> None:
    findings = cpl.scan(_read("main_excerpt.log"))

    # Read off the log by hand: the table and figure floats name their own generated
    # file; the two prose overflows belong to results.tex, which inputs those floats and
    # is open again once each float's file has closed.
    assert _summary(findings) == [
        ("overfull", 7.88, "sections/results_table.tex"),
        ("float-too-large", 84.87, "sections/results_figure_distribution.tex"),
        ("overfull", 9.31, "sections/results.tex"),
        ("overfull", 37.03, "sections/results.tex"),
        ("overfull", 124.04, "sections/model_summary.tex"),
    ]


CLEAN_LOG = "(./main.tex\n(./sections/narrow.tex)\n[1] (./main.aux) )\n"


def _paper(tmp_path: Path, main: str, supplementary: str) -> Path:
    (tmp_path / "main.log").write_text(main, encoding="latin-1")
    (tmp_path / "supplementary.log").write_text(supplementary, encoding="latin-1")
    return tmp_path


def test_cli_lists_each_problem_by_document_and_source_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paper = _paper(tmp_path, _read("main_excerpt.log"), CLEAN_LOG)

    assert cpl.main(["--paper", str(paper)]) == 1

    out = capsys.readouterr().out
    assert "main.log" in out
    assert "124.04pt too wide" in out and "sections/model_summary.tex" in out
    assert "84.87pt too tall" in out and "sections/results_figure_distribution.tex" in out
    assert "5 layout problem(s)" in out


def test_cli_passes_when_both_documents_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paper = _paper(tmp_path, CLEAN_LOG, CLEAN_LOG)

    assert cpl.main(["--paper", str(paper)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_fails_when_a_document_has_not_been_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "main.log").write_text(CLEAN_LOG, encoding="latin-1")

    assert cpl.main(["--paper", str(tmp_path)]) == 2
    assert "supplementary.log" in capsys.readouterr().out


def test_ignores_overflows_of_one_point_or_less() -> None:
    log = "\n".join(
        [
            "(./sections/supp/a.tex",
            "Overfull \\hbox (0.99998pt too wide) in paragraph at lines 5--17",
            " [][] ",
            "",
            "Overfull \\hbox (1.0pt too wide) in paragraph at lines 20--21",
            " []",
            "",
            "Overfull \\hbox (1.47514pt too wide) in paragraph at lines 5--17",
            " [][] ",
            "",
            ")",
        ]
    )

    assert _summary(cpl.scan(log)) == [("overfull", 1.48, "sections/supp/a.tex")]


def test_reports_a_landscape_float_taller_than_its_rotated_page() -> None:
    r"""``rotating`` zeroes a sideways float's box before turning it, so pdflatex never says
    ``Float too large`` for one. The generators set its content in a ``\vbox`` exactly as tall
    as the rotated page, so an overflow surfaces as an ``Overfull \vbox`` (lines from a real
    pdflatex log)."""
    log = "\n".join(
        [
            "(./sections/supp/exposure_matrix.tex",
            "",
            "Overfull \\vbox (169.29735pt too high) detected at line 12",
            " []",
            "",
            "Overfull \\vbox (0.5pt too high) detected at line 30",
            " []",
            "",
            ")",
        ]
    )

    assert _summary(cpl.scan(log)) == [
        ("overfull-vbox", 169.30, "sections/supp/exposure_matrix.tex")
    ]


def test_unwraps_file_names_split_across_79_character_log_lines() -> None:
    findings = cpl.scan(_read("wrapped_at_79.log"))

    folder = "sections/supp_very_long_directory_name_to_force_wrapping"
    assert _summary(findings) == [
        ("overfull", 93.32, f"{folder}/wide_table_with_a_long_file_name.tex"),
        ("float-too-large", 155.81, f"{folder}/tall_figure_with_a_long_file_name.tex"),
    ]
