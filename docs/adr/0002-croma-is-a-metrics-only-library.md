# croma is a metrics-only library; benchmarking lives elsewhere

`croma` ships as a lean, metrics-only Python library. General benchmarking is the
job of a separate package; the `scripts/` in this repo are paper-reproduction
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
  Committed to git but excluded from the wheel. The generators are public; the
  paper `.tex` source stays local (arXiv is its home). See ADR-0003 for what is
  and isn't committed.
- **Extras:** the `[bench]` extra is renamed `[repro]` — the deps needed to
  regenerate the paper from a clone, not to run a general benchmark.

## Distribution name

Import name is `croma`. The bare `croma` distribution on PyPI is held by an
unrelated dormant project; acquisition is being pursued. Until then the dist name
is a single, trivially reversible pyproject line with free brand-aligned fallbacks
(`croma-metrics`, etc.). Not worth blocking the release.
