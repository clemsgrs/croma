r"""Report every float that does not fit the page, from the logs ``paper/build.sh`` leaves.

CI has no TeX, so nothing there sees a table run past the margin or a figure taller than
the page. Generated tables grow rows and columns with the encoder panel, so the paper can
stop fitting without anyone touching a ``.tex`` file -- the 27-encoder panel built with 20
such floats. This is the local check that makes that visible, like ``check_paper_figures.py``:

  overfull          an ``Overfull \hbox`` wider than 1 pt -- a table or line past the margin
  float-too-large   a ``Float too large for page`` -- a float taller than the text block
  overfull-vbox     an ``Overfull \vbox`` higher than 1 pt -- in practice a landscape table
                    taller than its rotated page. ``rotating`` zeroes a sideways float's box
                    before turning it, so pdflatex never reports one as too large; the
                    generators set its content in a ``\vbox`` as tall as the rotated page
                    (``_paper_tables.landscape_table``) so that the overflow is reported here.

Each finding names the source file pdflatex was reading at the time, recovered from the
log's ``(file ... )`` nesting. A file's own name is what to fix: a float in a generated file
(``sections/supp/*.tex``) is fixed in its generator, never by hand (ADR-0010). An overflow
attributed to a hand-written file (``sections/results.tex``) is that file's own prose.

TeX Live writes logs wrapped at 79 characters by default, which splits long file names
across lines; such a log is unwrapped before it is read. TinyTeX, which this repo builds
with, sets a wide ``max_print_line`` and does not wrap.

Run after ``paper/build.sh``:
  python scripts/repro/check_paper_layout.py
Exits 1 when any finding is reported.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "paper"
#: The two documents ``paper/build.sh`` builds; each leaves ``<name>.log`` in ``paper/``.
DOCUMENTS = ("main", "supplementary")

#: An overflow at or under this is invisible in print and is not reported.
THRESHOLD_PT = 1.0
#: TeX Live's default ``max_print_line``: a wrapped log breaks its lines at this length.
WRAP = 79
_STATISTICS = "Here is how much of TeX's memory you used:"

_OVERFULL = re.compile(r"^Overfull \\(?P<box>[hv])box \((?P<pt>[\d.]+)pt too (?:wide|high)\)")
_UNDER_OR_OVERFULL = re.compile(r"^(?:Overfull|Underfull|Tight|Loose) \\[hv]box ")
_FLOAT = re.compile(r"Float too large for page by (?P<pt>[\d.]+)pt")
#: A ``(`` that opens a file: a path with an extension, or one that starts with ``./``/``/``.
_FILE = re.compile(r"\((?P<path>(?:\.{0,2}/)?[^\s()\[\]{}<>]*\.[A-Za-z0-9]+)")


@dataclass(frozen=True)
class Finding:
    """One float or line that does not fit."""

    kind: str  # "overfull", "float-too-large" or "overfull-vbox"
    points: float
    source: str  # the file pdflatex had open, as the log names it, without a leading ./
    message: str  # the log line that reported it


def _clean(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def scan(text: str) -> list[Finding]:
    r"""Every over-threshold ``Overfull \hbox``/``\vbox`` and every ``Float too large`` in a log.

    The open file is tracked by the log's parentheses: ``(path`` pushes a file, a bare ``(``
    pushes a placeholder so that its ``)`` pops it, and ``)`` pops. The box contents TeX
    prints after an over- or underfull box warning are typeset material, not log syntax --
    a table cell like ``MSKCC (TCGA-2x2`` would unbalance the count -- so they are skipped
    up to the blank line that ends them.
    """
    stack: list[str | None] = []
    findings: list[Finding] = []
    lines = unwrap(text.splitlines())
    i = 0
    while i < len(lines):
        line = lines[i]
        if _UNDER_OR_OVERFULL.match(line):
            match = _OVERFULL.match(line)
            if match and float(match["pt"]) > THRESHOLD_PT:
                kind = "overfull" if match["box"] == "h" else "overfull-vbox"
                findings.append(Finding(kind, float(match["pt"]), _current(stack), line))
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        match = _FLOAT.search(line)
        if match:
            findings.append(Finding("float-too-large", float(match["pt"]), _current(stack), line))
        _track(line, stack)
        i += 1
    return findings


def unwrap(lines: list[str]) -> list[str]:
    """Rejoin the lines pdflatex split at ``WRAP`` characters, if this log was wrapped.

    A wrapped log is recognised by having no longer line between its banner (the first
    line, written before the width applies) and its closing memory statistics, which are
    exempt too. An unwrapped log keeps its lines as they are: in one, a line that happens
    to be exactly ``WRAP`` long is a whole line -- TinyTeX's own log ends
    ``... (./sections/results.tex`` at exactly 79 characters -- and joining it to the next
    would corrupt the file name.
    """
    body = lines[1:]
    for n, line in enumerate(body):
        if line.startswith(_STATISTICS):
            body = body[:n]
            break
    if any(len(line) > WRAP for line in body):
        return lines
    joined: list[str] = []
    carry = ""
    for line in lines:
        if len(line) == WRAP:
            carry += line
            continue
        joined.append(carry + line)
        carry = ""
    if carry:
        joined.append(carry)
    return joined


def _current(stack: list[str | None]) -> str:
    for entry in reversed(stack):
        if entry is not None:
            return _clean(entry)
    return "?"


def _track(line: str, stack: list[str | None]) -> None:
    pos = 0
    while pos < len(line):
        char = line[pos]
        if char == "(":
            match = _FILE.match(line, pos)
            if match:
                stack.append(match["path"])
                pos = match.end()
                continue
            stack.append(None)
        elif char == ")" and stack:
            stack.pop()
        pos += 1


def _describe(finding: Finding) -> str:
    if finding.kind == "overfull":
        return f"Overfull \\hbox, {finding.points:.2f}pt too wide"
    if finding.kind == "overfull-vbox":
        return f"Overfull \\vbox, {finding.points:.2f}pt too tall"
    return f"Float too large, {finding.points:.2f}pt too tall"


def _where(finding: Finding) -> str:
    """The source lines pdflatex names, e.g. ``lines 5--49`` or ``input line 11``."""
    match = re.search(r"(?:input )?lines? [\d-]+", finding.message)
    return match[0] if match else finding.message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--paper", type=Path, default=PAPER, help="the built manuscript tree (default: paper/)"
    )
    args = parser.parse_args(argv)

    missing = [doc for doc in DOCUMENTS if not (args.paper / f"{doc}.log").is_file()]
    if missing:
        names = ", ".join(f"{doc}.log" for doc in missing)
        print(f"{names} not found under {args.paper}: run paper/build.sh first.")
        return 2

    total = 0
    for doc in DOCUMENTS:
        log = args.paper / f"{doc}.log"
        findings = scan(log.read_text(encoding="latin-1"))
        total += len(findings)
        if findings:
            print(f"\n{log.name} ({len(findings)}):")
            for finding in findings:
                print(f"  {_describe(finding):40s} {finding.source}  ({_where(finding)})")

    logs = ", ".join(f"{doc}.log" for doc in DOCUMENTS)
    if total:
        print(f"\n{total} layout problem(s) across {logs}.")
        return 1
    print(
        f"OK: no Overfull \\hbox or \\vbox over {THRESHOLD_PT:g}pt and no Float too large "
        f"in {logs}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
