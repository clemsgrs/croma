"""Report which hand-copied figures under ``paper/figures/`` have fallen behind ``output/``.

Nothing in this repo stages figures into the manuscript tree. Figure scripts render beside
the data they read (``output/.../plots/{pdf,png}/``) and a human copies the ones a float
earns. That is deliberate: the script that used to stage them
(``regen_paper_figs_faithful.py``) pinned a run directory, silently broke when the tile panel
moved from ``k-star`` to ``median-k``, and left its copies months behind the run they claimed
to show. Nobody noticed, because a figure -- unlike a table -- had no freshness test
(ADR-0010 covers ``paper/sections/`` only).

Manual curation is the right call, but it needs the drift to be *visible*. This script only
looks and prints. It never copies, never deletes, never writes:

  MISSING     a document ``\\includegraphics`` it, but no such file exists -> compiles to a
              LaTeX error. If a rendered artifact of that name exists, it is named as a hint.
  STALE       the artifact this figure was copied from is newer -> re-copy to pick up the
              current run.
  UNSOURCED   nothing under ``output/`` renders this name. Either hand-drawn (TikZ,
              Illustrator) or its generator no longer runs.
  UNUSED      present in ``paper/figures/``, cited by no document -> a leftover, safe to
              delete by hand.

Two documents live in ``paper/``: ``main.tex`` (the manuscript) and ``draft.tex`` (a longer
working draft). They cite different figures, so both are walked and every finding names the
documents that cite it -- a figure stale only in ``draft.tex`` is not a submission blocker.

A figure's source is resolved by *run*, never by name alone: every benchmark emits its own
``knn_bio_k_sweep.pdf``, so a bare basename match would compare Camelyon's copy against
PANDA's. ``figures/results/<benchmark>[-suffix]/pdf/x.pdf`` is matched against that
benchmark's run (via ``paper_manifest``, so the protocol is never spelled out here) and then
against any standalone study under ``output/studies/``.

Run:  python scripts/repro/check_paper_figures.py [--strict]
      --strict exits 1 when anything is MISSING or STALE (for a pre-submission gate).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from paper_manifest import by_benchmark  # noqa: E402

PAPER = REPO / "paper"
FIGROOT = PAPER / "figures"
OUTPUT = REPO / "output"

EXTENSIONS = (".pdf", ".png")

_INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_INPUT = re.compile(r"\\input\{([^}]+)\}")
#: An unescaped ``%`` starts a LaTeX comment; a commented-out figure is not cited.
_COMMENT = re.compile(r"(?<!\\)%.*")


def _stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _documents() -> list[Path]:
    """Top-level ``paper/*.tex`` that declare a document class, i.e. real build targets."""
    return sorted(p for p in PAPER.glob("*.tex") if r"\documentclass" in p.read_text())


def _reachable(document: Path) -> list[Path]:
    r"""The ``.tex`` files ``document`` compiles, by walking ``\input``.

    Not every ``.tex`` under ``paper/``: a retired draft in ``sections/kstar-backup/`` and the
    standalone ECDP abstract in ``ecdp/`` both cite figures, and counting them would let a
    file nothing compiles keep a deleted figure marked "used". ``\input`` takes the extension
    or leaves it off, and both forms compile.
    """
    seen: list[Path] = []

    def walk(path: Path) -> None:
        if not path.exists() or path in seen:
            return
        seen.append(path)
        for rel in _INPUT.findall(_COMMENT.sub("", path.read_text())):
            walk(PAPER / (rel if rel.endswith(".tex") else f"{rel}.tex"))

    walk(document)
    return seen


def _cited() -> dict[Path, set[str]]:
    r"""Every compiled ``\includegraphics``, resolved to a path, mapped to its documents.

    ``\graphicspath{{./}{figures/}}``, compiled from ``paper/``, means a cited path is tried
    relative to ``paper/`` then to ``paper/figures/``; an extensionless citation is tried
    against each extension, as LaTeX would. A path built from a macro (``\plotdir/x.png``,
    used only by the standalone ECDP abstract) cannot be resolved statically and is skipped.
    A citation that resolves nowhere is keyed by its ``paper/figures/`` candidate: that is
    where a human would copy it to.
    """
    cited: dict[Path, set[str]] = {}
    for document in _documents():
        for tex in _reachable(document):
            for rel in _INCLUDE.findall(_COMMENT.sub("", tex.read_text())):
                if "\\" in rel:
                    continue
                suffixes = [""] if Path(rel).suffix else list(EXTENSIONS)
                candidates = [root / f"{rel}{ext}" for ext in suffixes for root in (PAPER, FIGROOT)]
                resolved = next((c for c in candidates if c.exists()), FIGROOT / f"{rel}{suffixes[0]}")
                cited.setdefault(resolved, set()).add(document.name)
    return cited


def _study_roots() -> list[Path]:
    return sorted(OUTPUT.glob("studies/*/plots/pdf"))


def _benchmark_roots(figure: Path) -> list[Path]:
    """Plot directories of the run whose name the figure's parent directory carries.

    ``figures/results/pathorob-camelyon/pdf/x.pdf`` -> that benchmark's run plots and its
    studies' plots. The run directory itself comes from ``paper_manifest``, so this never
    names a protocol.

    The directory must be named for the benchmark exactly. It used to be allowed a
    ``-faithful``/``-reduced`` suffix, stripped here before the lookup, because the paper's
    figure tree was once keyed by *manifest* rather than by run -- and ``-faithful`` in
    particular named no run at all. Both directories are gone (one renamed, one deleted),
    and the stripping went with them: it let a directory keep a name its run did not have,
    which is what allowed the figures inside to sit months behind that run unnoticed. An
    unrecognised name now falls through to UNSOURCED, loudly, instead of quietly resolving.
    """
    parts = figure.parts
    if figure.parent.name != "pdf" or "results" not in parts:
        return []
    try:
        run = REPO / by_benchmark(figure.parent.parent.name).run_rel
    except ValueError:  # not a benchmark this paper reports
        return []
    return [run / "plots" / "pdf", run / "studies" / "plots" / "pdf"]


def _sources_for(figure: Path) -> list[Path]:
    roots = _benchmark_roots(figure) + _study_roots()
    return [candidate for root in roots if (candidate := root / figure.name).exists()]


def _rendered_index() -> dict[str, list[Path]]:
    """Every rendered PDF by name -- only to hint where a MISSING figure could come from."""
    index: dict[str, list[Path]] = {}
    for path in OUTPUT.rglob("plots/pdf/*.pdf"):
        index.setdefault(path.name, []).append(path)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any figure is MISSING or STALE")
    args = parser.parse_args()

    if not FIGROOT.is_dir():
        print("paper/figures/ absent (paper/ is git-ignored); nothing to check.")
        return 0

    cited = _cited()
    rendered = _rendered_index()
    on_disk = sorted(p for p in FIGROOT.rglob("*.pdf")
                     if ".ipynb_checkpoints" not in p.parts)

    missing = sorted((p for p in cited if not p.exists()), key=lambda p: p.name)
    stale: list[tuple[Path, Path]] = []
    unsourced: list[Path] = []
    for figure in on_disk:
        sources = _sources_for(figure)
        if not sources:
            unsourced.append(figure)
            continue
        newest = max(sources, key=lambda p: p.stat().st_mtime)
        if newest.stat().st_mtime > figure.stat().st_mtime:
            stale.append((figure, newest))
    unused = [f for f in on_disk if f not in cited]

    def who(path: Path) -> str:
        return ", ".join(sorted(cited.get(path, {"-"})))

    if missing:
        print(f"\nMISSING -- cited, but no such file ({len(missing)}):")
        for path in missing:
            print(f"  {_rel(path)}   [cited by {who(path)}]")
            for source in rendered.get(path.name, []):
                print(f"      render exists, copy it: {_rel(source)}")

    if stale:
        print(f"\nSTALE -- the artifact it was copied from is newer ({len(stale)}):")
        for figure, source in sorted(stale, key=lambda s: s[0].name):
            print(f"  {figure.name:32s} paper {_stamp(figure)}  <  {_stamp(source)}   "
                  f"[cited by {who(figure)}]")
            print(f"      source: {_rel(source)}")

    if unsourced:
        print(f"\nUNSOURCED -- nothing under output/ renders these ({len(unsourced)}):")
        print("  (expected for hand-drawn figures; otherwise the generator no longer runs)")
        for path in unsourced:
            print(f"  {_rel(path):72s} [cited by {who(path)}]")

    if unused:
        print(f"\nUNUSED -- present in paper/figures/, cited by no document ({len(unused)}):")
        for path in unused:
            print(f"  {_rel(path)}")

    print(f"\n{len(missing)} missing, {len(stale)} stale, {len(unsourced)} unsourced, "
          f"{len(unused)} unused across {', '.join(d.name for d in _documents())}. "
          f"Reports only; nothing was copied.")

    return 1 if (args.strict and (missing or stale)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
