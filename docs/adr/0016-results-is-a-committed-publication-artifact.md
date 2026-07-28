# `results/` is a committed publication artifact

The documentation site publishes benchmark numbers. Those numbers reach it through
`results/`: a small, tracked, curated set of CSVs at the repository root, exported
from `output/` by a tracked script, carrying a provenance sidecar and guarded by a
freshness test.

`output/` is unchanged — still git-ignored in full, still safe to wipe.

## Why

The docs build runs `sphinx -W` on a clean CI checkout. It cannot see `output/`
(git-ignored, ADR-0007), `paper/` (git-ignored, ADR-0003) or `scripts/repro/`
(git-ignored, ADR-0012). Anything the site shows must therefore already be committed.
That is not a preference; it is the only shape a published number can take.

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

**Why the exporter is tracked, when `scripts/repro/` is not.** ADR-0012 keeps the paper
generators local because "publishing the build system for a document nobody can read is
the worst of both worlds: a reader cannot check the generators against the artifacts
they produce, because the artifacts are absent." That reasoning does not carry over
here — it inverts. The artifact *is* present: `results/*.csv` is committed and rendered
on a public site. Publishing the exporter is what makes the numbers checkable rather
than asserted. Keeping it local would leave `results/` in the public repo as data with
no visible derivation, which is the failure ADR-0012 was avoiding, not an instance of
the rule it set.

## Shape

**Tracked:**

- `results/<benchmark>.csv` — one per published cohort (`camelyon`, `tcga-4x4`,
  `tolkach-esca`), holding only the columns the site publishes.
- `results/cross_benchmark.csv` — the aggregate the front page and README render.
- `results/PROVENANCE.json` — protocol, per-cohort `k`, `tau` policy, roster size,
  `croma` version, source run path, export date, and a sha256 per committed CSV.
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
- `paper/` and `scripts/repro/` — ADR-0003 and ADR-0012 are untouched. The paper's
  `.tex` tables and this exporter render overlapping numbers through separate paths;
  they are not unified, because unifying them would drag the local-only tree into the
  public build.

## Consequences

- **The published scope is narrower than the paper's.** `results/` carries the three
  tile cohorts with a shared 21-model roster. Prostate (16 models) and PANDA
  (slide-level, n=4) are computed and stay in `output/`, unpublished, because a site
  table with a silently different roster misleads more than it informs.
- **Two paths can now disagree.** The paper's tables come from `scripts/repro/`, the
  site's from `scripts/tools/`. The freshness test pins the site's path to `output/`;
  it does not compare the two. If a divergence ever matters, the fix is a cross-check
  test, not a merged generator.
- **`results/` ships in the sdist.** Hatchling includes everything not VCS-ignored, so
  the published CSVs and SVGs travel with the wheel's source distribution. They are a
  few tens of KB and describe the package's own evaluation, so this is acceptable
  rather than merely tolerated.
- **Republishing is an explicit step.** Re-running a benchmark does not update the site.
  Someone must run the exporter and commit the diff — which is the point: the diff is
  the review surface for changing a public claim.

## When this changes

If the manuscript is published and `paper/` becomes tracked (the reversal ADR-0012
describes), the paper tables and `results/` become two views of one tracked tree, and
merging the generators becomes worth considering. Until then, the duplication is the
price of publishing numbers whose LaTeX assembly is not public.
