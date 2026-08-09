# croma is a lean metrics library; benchmarking lives elsewhere

> **Narrowed by [ADR-0011](0011-downstream-shortcut-susceptibility-ships-in-the-library.md).**
> `croma.downstream` ships the confounder-biased probe protocol and both of its
> reductions, so the library is no longer *metrics-only*. The constraint that survives is
> leanness: `croma.downstream` is numpy/sklearn-only and adds zero install weight, and
> torch stays confined to the `[repro]` extra. General multi-model benchmarking is still
> out of scope.
>
> **See also [ADR-0012](0012-paper-tooling-stays-local.md)** for the current status of
> `scripts/` — paper generators are tracked in the repository but excluded from distributions.

`croma` ships as a lean Python library of robustness metrics. General benchmarking is
the job of a separate package; the `scripts/` in this repo are paper-reproduction
tooling, not a shipped product.

## Why

The contribution is the metrics (RI, MaRI, CRoMa) and their reporting. A separate
package already owns general multi-model benchmarking, so folding a pipeline into
`croma` would duplicate it and bloat the install. Keeping `croma` small makes it
easy to depend on, test, and keep backward-compatible.

This also fixes a real defect: the old `[bench]` extra installed heavy deps
(torch, timm, transformers) but the code that used them lived in `scripts/`, which
was never in the wheel. `pip install ...[bench]` gave dependencies and no pipeline.

## Shape

- **Package (shipped):** `src/croma/` — src-layout, so tests run against the
  installed package and missing-file-in-wheel bugs surface immediately.
- **Public API (leanest):** `from croma import RI, MaRI, CRoMa,
  expand_features_to_manifest, __version__`. Full class names and result types
  (`CRoMaResult`, `RobustnessResult`) are reachable via `croma.metrics.*` but are
  not part of the stable top-level contract. Minimal-first: promote on demand,
  since removing a public name later is breaking. LTM/tail statistics ride on
  `CRoMaResult`, not as separate exports.
- **`scripts/` (repo-only):** reproduces *this paper* — it writes
  `paper/sections/generated_values.tex`, the result tables, and the figures.
  Excluded from the wheel. The paper `.tex` source stays local (arXiv is its home),
  and per ADR-0012 the generators that render into it are tracked but excluded from source
  distributions; the benchmarking pipeline and studies remain committed and distributed.
  See ADR-0003 and ADR-0012 for what is and isn't committed.
- **Extras:** the `[bench]` extra is renamed `[repro]` — the deps needed to
  regenerate the paper from a clone, not to run a general benchmark.

## Distribution name

Import name and distribution name are both `croma`. **Resolved:** the name is ours on
PyPI — the project exists there with zero files, which `0.1.0` claims. The
`croma-metrics` fallback contemplated here is not needed and was never used.

The project was briefly distributed as `cross-margin` 1.0.0, before the metric was
renamed (ADR-0001) and before `tau` was resolved automatically. That name is retired and
no code is carried forward from it.
