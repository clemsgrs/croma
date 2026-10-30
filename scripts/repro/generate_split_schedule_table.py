r"""Render the confounder-biased probe schedules (``supp/split_schedule.tex``, ``tab:split-schedule``).

The downstream protocol was described only by reference to PathoROB. This table states it:
per benchmark, the cohort the probe is trained and tested on and the held-out centres it is
only tested on; per split, how biased the training set is and how large each of its slices
is. Every number comes from ``_split_schedule``, which runs the study's own schedule and the
library sweep's own slicing over the APD metadata, so nothing here is arithmetic of its own.

The probe itself is the vendored PathoROB one; its settings are stated once, in the
caption. ``C_GRID`` restates the regularisation grid that code builds inline, and a test
holds the two together.

``--tiles-per-slide camelyon=30`` (repeatable) renders the same table for a reduced-tile
variant of the sweep, under its own ``--label`` and ``--out``.

Usage:
  python scripts/repro/generate_split_schedule_table.py
  python scripts/repro/generate_split_schedule_table.py --tiles-per-slide camelyon=30 \
      --label tab:split-schedule-30 --out paper/sections/supp/split_schedule_30.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from _split_schedule import REPO, Schedule, schedules

OUT = REPO / "paper/sections/supp/split_schedule.tex"
LABEL = "tab:split-schedule"

#: The probe's regularisation grid as ``np.linspace`` arguments over log10 C: 15 values from
#: 1e-8 to 1e4. Restated from the vendored ``train_logistic_regression``, which builds it
#: inline; ``tests/test_split_schedule_table.py`` reads that source to keep the two equal.
C_GRID = (-8, 4, 15)

FOOTNOTE = "a"


def _n(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _centre(name: str) -> str:
    """A centre as the paper writes it: Tolkach-ESCA's cohort codes without their prefix."""
    name = re.sub(r"^VALSET\d+_", "", name).removesuffix("_FULL")
    if name.islower():
        name = name.capitalize()
    return name.replace("&", r"\&").replace("_", r"\_")


def _name(s: Schedule) -> str:
    return s.name + (rf"$^{{{FOOTNOTE}}}$" if s.croma_authored else "")


def _pool(pool, width: str) -> str:
    centres = ", ".join(_centre(c) for c in pool.centres)
    # A slide-level pool holds one embedding per slide; its tile count would repeat it.
    tiles = "" if pool.tiles == pool.slides else f", {_n(pool.tiles)} tiles"
    return rf"\parbox[t]{{{width}}}{{\raggedright {centres} ({_n(pool.slides)} slides{tiles})}}"


def _protocol_panel(rows: list[Schedule]) -> list[str]:
    lines = [
        r"\begin{tabular}{lcclll}",
        r"\hline",
        r"\multicolumn{6}{l}{\emph{(a) Cohorts and replicates}} \\",
        r"\hline",
        r"Benchmark & Tiles/slide & $K$ & ID pool & OOD pool & Replicates (seed) \\",
        r"\hline",
    ]
    for s in rows:
        lines.append(
            f"{_name(s)} & {s.tiles_per_slide} & {s.n_classes} & "
            f"{_pool(s.id_pool, '4.3cm')} & {_pool(s.ood_pool, '4.3cm')} & "
            rf"{s.replicates} ({s.seed}) \\[2pt]"
        )
    return lines + [r"\hline", r"\end{tabular}"]


def _favourable(values: tuple[int, ...]) -> str:
    return "/".join(str(v) for v in values)


def _test(slides: tuple[int, int]) -> str:
    low, high = slides
    return str(low) if low == high else f"{low}--{high}"


def _split_panel(rows: list[Schedule]) -> list[str]:
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"\multicolumn{8}{l}{\emph{(b) Splits}} \\",
        r"\hline",
        r" & & & \multicolumn{2}{c}{Train slides/cell} & & & \\",
        r"\cline{4-5}",
        r"Benchmark & Split & Cramér's $V$ & Fav. & Unfav. & Training & Validation & "
        r"Test slides/cell \\",
        r"\hline",
    ]
    for s in rows:
        for split in s.splits:
            name = _name(s) if split.index == 0 else ""
            lines.append(
                f"{name} & {split.index} & {split.cramers_v:.3f} & "
                f"{_favourable(split.favourable)} & {split.unfavourable} & "
                f"{_n(split.train)} & {_n(split.validation)} & {_test(split.test_slides)} \\\\"
            )
        lines.append(r"\hline")
    return lines + [r"\end{tabular}"]


_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _types_sentence(rows: list[Schedule]) -> str:
    return "".join(
        f" On {s.name} the favourable cells shrink on {_WORDS[len(s.favourable_types)]} "
        f"trajectories, listed as {'/'.join(s.favourable_types)}."
        for s in rows
        if len(s.favourable_types) > 1
    )


def _case_sentence(rows: list[Schedule]) -> str:
    cased = [s for s in rows if s.case_based]
    return "".join(
        f" On {s.name} the independence unit is the case: panel~a counts cases, and a "
        f"slide in panel~b is one case's {s.tiles_per_slide}-tile block within a class, since "
        "a case contributes tiles to several classes."
        for s in cased
    )


def _slide_level_sentence(rows: list[Schedule]) -> str:
    names = [s.name for s in rows if s.tiles_per_slide == 1]
    if not names:
        return ""
    return (
        f" {' and '.join(names)} is slide-level, with one embedding per slide, so its "
        "training and validation sizes count slides."
    )


def _caption(rows: list[Schedule]) -> str:
    low, high, n = C_GRID
    return (
        r"\textbf{Confounder-biased probe schedules.} Each schedule walks the in-domain "
        r"training set from a balanced centre$\times$class composition (split~0, Cramér's "
        r"$V{=}0$) to a fully confounded one ($V{=}1$) at a constant training size, while "
        r"the held-out test sets stay fixed. \textbf{a,} Per benchmark: tiles per slide; the "
        r"number of biological classes $K$ (chance $\pi{=}1/K$); the in-domain (ID) centres "
        r"the probe is trained, validated and tested on, and the out-of-domain (OOD) "
        r"centres it is only tested on; and the number of replicates and their seed. "
        r"\textbf{b,} Per split: Cramér's $V$ of the training centre$\times$class counts; "
        r"training slides per favourable (shortcut-opposing) and unfavourable "
        r"(shortcut-aligned) cell; the training and validation set sizes in tiles; and the "
        r"ID test slides per cell." + _types_sentence(rows) + _case_sentence(rows)
        + _slide_level_sentence(rows)
        + r" The probe is an L2-regularised logistic regression with scikit-learn defaults. "
        rf"Its inverse regularisation strength is chosen from {n} log-spaced values in "
        rf"$[10^{{{low}}}, 10^{{{high}}}]$ by validation balanced accuracy, and the probe is "
        r"refitted on the training set at that value. Every test set is scored by balanced "
        r"accuracy."
    )


def _footnote(rows: list[Schedule]) -> str:
    notes = []
    for s in rows:
        if not s.croma_authored:
            continue
        load = s.splits[0].unfavourable
        step = s.splits[1].unfavourable - load
        deviations = [
            r"the schedule is our own: Camelyon's two-centre diagonal design, "
            rf"re-parameterised to a balanced load of {load} slides per cell and a step of "
            rf"{step} slides per split."
        ]
        if s.validation_fraction is not None:
            deviations.append(
                rf"the sweep is given an explicit validation fraction of "
                rf"${s.validation_fraction:g}$ of each cell's training slides: PathoROB's "
                r"rule reserves $1/\max(\text{training slides})$ of a cell's training rows, "
                r"which rounds to zero rows at one embedding per slide."
            )
        ordinals = ["First", "Second"]
        notes.append(
            rf"$^{{{FOOTNOTE}}}$PathoROB defines no schedule for {s.name}, and its protocol "
            rf"here deviates from PathoROB's in {_WORDS[len(deviations)]} documented "
            f"way{'s' if len(deviations) > 1 else ''}. "
            + " ".join(
                f"{ordinals[i]}, {text}" if len(deviations) > 1 else text[0].upper() + text[1:]
                for i, text in enumerate(deviations)
            )
        )
    return " ".join(notes)


def build(
    root: Path = REPO,
    *,
    tiles_per_slide: Mapping[str, int] | None = None,
    label: str = LABEL,
) -> str:
    rows = schedules(root, tiles_per_slide=tiles_per_slide)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        *_protocol_panel(rows),
        "",
        r"\vspace{5pt}",
        *_split_panel(rows),
    ]
    note = _footnote(rows)
    if note:
        lines += ["", r"\vspace{3pt}", rf"\parbox{{\linewidth}}{{\footnotesize {note}}}"]
    lines += [
        rf"\caption{{{_caption(rows)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _override(text: str) -> tuple[str, int]:
    key, _, value = text.partition("=")
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected <benchmark>=<positive int>, got {text!r}")
    return key, int(value)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles-per-slide", type=_override, action="append", default=[])
    p.add_argument("--label", default=LABEL)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    tex = build(tiles_per_slide=dict(args.tiles_per_slide), label=args.label)
    if args.check:
        if not out.exists() or out.read_text() != tex:
            print(f"STALE  {out.relative_to(REPO)}", file=sys.stderr)
            return 1
        print(f"ok     {out.relative_to(REPO)}", file=sys.stderr)
        return 0
    if not out.parent.exists():
        print("skip   paper/ absent", file=sys.stderr)
        return 0
    out.write_text(tex)
    print(f"wrote  {out.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
