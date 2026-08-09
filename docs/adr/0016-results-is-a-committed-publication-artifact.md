# `results/` is a committed publication artifact

The documentation site publishes benchmark numbers. Those numbers reach it through
`results/`: a small, tracked, curated set of CSVs at the repository root, exported
from `output/` by a tracked script, carrying a provenance sidecar and guarded by a
freshness test.

`output/` is unchanged — still git-ignored in full, still safe to wipe.

## Why

The docs build runs `sphinx -W` on a clean CI checkout. It cannot see `output/`
(git-ignored, ADR-0007) or `paper/` (git-ignored, ADR-0003). The generators under
`scripts/repro/` are tracked after manuscript publication (ADR-0012), but their run inputs and
LaTeX outputs remain absent. Anything the site shows must therefore already be committed. That
is not a preference; it is the only shape a published number can take.

The alternative was to hand-write the tables into `.rst`. Rejected: a benchmark re-run
would leave the site asserting numbers no run produced, and nothing would detect it.
The repository has been here before — the 8→4 centre change silently invalidated every
stored TCGA CRoMa value, and the artifacts written before the fix were simply wrong.
A published table needs a mechanical link back to the run that produced it.

**Why `results/` is not under `output/`.** A `!output/results/` carve-out would keep
everything results-shaped under one root, and was rejected on lifecycle grounds. The
two trees change for opposite reasons: `output/` is regenerable scratch that any
benchmark run rewrites, while `results/` changes only on a deliberate republish, and
its git history is the record of what the public site has claimed over time. A tracked
island inside a tree that pipelines clear — and that `git clean -xdf` targets — is a
footgun for the sake of tidiness.

**Why the exporter lives outside the paper generators.** The artifact is public:
`results/*.csv` is committed and rendered on the documentation site. Its tracked exporter makes
the numbers checkable rather than asserted. It belongs under `scripts/tools/` because output
destination, not tracking status, defines script ownership (ADR-0006): it writes repository
publication artifacts, whereas `scripts/repro/` writes the ignored paper assembly.

## Shape

**Tracked:**

- `results/<benchmark>.csv` — one per published cohort (`camelyon`, `tcga-4x4`,
  `tolkach-esca`), holding only the columns the site publishes.
- `results/cross_benchmark.csv` — the aggregate the front page and README render.
- `results/distributions.json` — 200-bin histograms of per-sample CRoMa, per encoder and
  cohort. What the distribution explorer reads, and the reason the explorer needs no
  server: `html_extra_path` copies the whole tree to the site root.
- `results/PROVENANCE.json` — protocol, per-cohort `k`, `tau` policy, roster size,
  `croma` version, source run path, export date, and a sha256 per committed data file.
- A marked region of `README.md`, rewritten in place. A hand-written table in the README
  is the same hazard as a hand-written one on the site with less to catch it, so the
  exporter owns that block too. Its checksum is deliberately *not* in `PROVENANCE.json`:
  the file is mostly prose, and a hash over it would go stale on every wording change
  while saying nothing about the numbers. The freshness test covers it instead.
- `scripts/tools/export_results.py` — reads
  `output/metrics/<protocol>/<benchmark>/results/metrics.csv`, writes the above. In
  `tools/` rather than `repro/` because its output destination is the repository, not
  `paper/` (the grouping rule of ADR-0006).
- `tests/test_results_export.py` — regenerates `results/` from `output/` and asserts
  equality plus checksum agreement. `skipif` when `output/` is absent, so it guards on
  a machine that holds the runs and no-ops in CI. This is deliberately the *guarded*
  pattern ADR-0012 found missing from the paper builder cases, not the unguarded one.
- `docs/_static/figures/*.svg` — published figures as transparent light/dark pairs,
  selected by furo's `only-light` / `only-dark`. Figures are artifacts on exactly the
  same footing as the CSVs: re-rendered for the web, committed, never fetched at build
  time.

**Still git-ignored, unchanged:**

- `output/` in full — scratch, wipe-safe, the single source the exporter reads.
- `paper/` — ADR-0003 and ADR-0012 are untouched. The tracked paper generators under
  `scripts/repro/` remain excluded from distributions. Their `.tex` tables and this exporter
  render overlapping numbers through separate paths; they are not unified, because unifying
  them would drag paper-only assembly into the public build.

## Consequences

- **The published scope is narrower than the paper's.** `results/` carries the three
  tile cohorts with a shared 21-model roster. Prostate (16 models) and PANDA
  (slide-level, n=4) are computed and stay in `output/`, unpublished, because a site
  table with a silently different roster misleads more than it informs.
- **Two paths can now disagree.** The paper's tables come from `scripts/repro/`, the
  site's from `scripts/tools/`. The freshness test pins the site's path to `output/`;
  it does not compare the two. If a divergence ever matters, the fix is a cross-check
  test, not a merged generator.
- **`results/` ships in the sdist.** The sdist is an explicit `include` list, so this is
  a deliberate entry rather than a consequence of hatchling's defaults. It has to be
  there: `tests/test_results_export.py` reads the committed tree, and an sdist without
  it would ship a test suite that cannot run. A few tens of KB describing the package's
  own evaluation is acceptable rather than merely tolerated.
- **Republishing is an explicit step.** Re-running a benchmark does not update the site.
  Someone must run the exporter and commit the diff — which is the point: the diff is
  the review surface for changing a public claim.

## When this changes

If the paper assembly itself becomes tracked, the paper tables and `results/` become two views
of one tracked tree, and merging the generators becomes worth considering. Until then, the
duplication is the price of publishing numbers whose LaTeX assembly is not in the repository.
