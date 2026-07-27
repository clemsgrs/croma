# The probe sweep takes a per-replicate slide arrangement

`probe_sweep_over_test_sets` accepts an `arrange_slides` callable: how one replicate
orders each `(confounder, class)` cell's slides before the schedule is cut out of them.
Its default is the sweep's own shuffle, so the reference protocol is unchanged. It is
*not* offered on `croma.probe_sweep`, the promoted entry point.

## Why a hook exists at all

ADR-0011 put the protocol in the library so that the paper's downstream numbers are
produced by the code a reader installs. Four of the five cohorts the APD study runs went
onto `probe_sweep` unchanged. The fifth did not.

PathoROB's Tolkach-ESCA cohort annotates patches by *case*, and a case contributes
patches to several biological classes at once. Training on a case in one class and
testing on it in another leaks, so upstream enumerates the case splits that avoid it;
each replicate draws one and moves those cases' slides to the tail the sweep tests from.
That is a property of one cohort's annotation, and #82 shipped the sweep without it,
recording in the module docstring that a cohort needing it was not served.

Leaving it there had a price the shipping slice made concrete: Tolkach would have kept a
*second* copy of the sweep in `scripts/studies/apd/` — the slicing, the validation take,
the held-out offset and the probe call — reaching into `croma.downstream._pathorob` for
the vendored trainer. One reduction with one implementation, but the protocol with two,
and the copy free to drift while every number still computed. The issue asked for the
study to import *the protocol*, not only the reductions.

## Why this shape

The seam is placed where upstream's own loop varies. PathoROB's driver is one loop with
`if dataset == "tolkach"` branches at exactly one point: after a cell is chunked into
slides and shuffled. Everything before and after is common to all cohorts. So the hook
does not invent a variation point, it names the one the reference implementation already
has.

What an arrangement receives is *slides* — `cells[confounder][class]` holding row indices
grouped by slide — not rows. A slide is the unit the protocol keeps whole, since its rows
are near-duplicates and splitting one across training and test scores the probe on tissue
it was fitted on. Handing over slides means an arrangement cannot break that however it
reorders them, and the sweep additionally refuses a return value that is not a
rearrangement of what it handed out: a lost, invented or broken-up slide would reshape
the cohort under a schedule written for it, and the sweep would still return a matrix.

It also receives the replicate's generator, because a faithful arrangement must consume
the same stream in the same order as the driver it reproduces — Tolkach draws its case
split *before* it shuffles, and swapping the two re-seeds every Tolkach number.

## Cost, against ADR-0002's minimal-first principle

The hook exposes the cell structure, which is otherwise internal, and that is a real
information leak: a caller can now see how the sweep groups rows. Three things bound it.

- The parameter sits on `probe_sweep_over_test_sets`, which ADR-0002's minimal-first
  principle already keeps off the top level, so it carries **no stability promise**. The
  promoted `croma.probe_sweep` signature is untouched.
- The default is the reference protocol, so nothing about a call that omits it changes.
- It is a *measurement* parameter, not pipeline machinery: it takes indices and a
  generator, loads nothing, reads nothing off disk. The ADR-0011 boundary — the library
  never learns where a repository keeps its files — holds.

## Rejected: keep the Tolkach path in the study

Defensible on the issue's own wording, since the two slide-level protocol deviations
already stay in the study as dataset configuration. Rejected because those two are
*arguments* to the sweep (a schedule, a validation fraction) while this one would have
been a second sweep: the same protocol, implemented twice, one copy of it unversioned,
untested and outside the package the paper tells readers to install.

## Consequences

- All five cohorts of the APD study run through one sweep, and the study layer holds no
  reduction and no protocol — only the arrangement Tolkach's annotation requires, which is
  ~10 lines of cell reordering handed to the library rather than a loop around it.
- The `croma.downstream` docstring no longer says Tolkach-ESCA is unserved.
- `croma.downstream.IN_DOMAIN` is exported, since a caller reading the multi-test result
  needs the key by name rather than by string literal.
