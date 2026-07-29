# Paper tooling stays local until the manuscript is published

Everything that renders into `paper/` is untracked: the `.tex` and figure generators
under `scripts/repro/`, the manifest mapping runs to paper artifacts, and their tests.
They remain on disk and keep working; they are simply not published.

This supersedes ADR-0003's "commit the generators" posture for as long as the
manuscript is unpublished. ADR-0003's *reasons* still hold and this is not a rejection
of them — see "When this reverses".

## Why

`paper/` was already git-ignored (ADR-0003: the manuscript belongs on arXiv, not
coupled to repo history). Publishing the build system for a document nobody can read
is the worst of both worlds: a reader cannot check the generators against the artifacts
they produce, because the artifacts are absent.

It also removed a defect rather than papering over one. `tests/test_paper_artifacts.py`
had eight builder cases that called the float builders directly instead of guarding on
`paper/` the way the freshness cases do. They read `output/metrics/`, which is
git-ignored and absent from any fresh checkout, so they raised `FileNotFoundError`
rather than skipping — and passed only on a machine that happened to hold the run
outputs. A clean checkout scored 8 failed / 382 passed; the first CI run on this branch
would have been red.

The deeper point is that the paper-artifact tests cannot be meaningfully public while
their inputs are not. The freshness guarantee is a local pre-flight by construction
(see the `tests/test_paper_artifacts.py` docstring), and a test suite whose every case
skips in CI is not a guarantee the repository can offer anyone else.

## Shape

**Local-only (on disk, git-ignored):**

- `scripts/repro/` — the `.tex` and figure generators, `paper_manifest.py`,
  `build_paper.py`, `check_paper_figures.py`, the templates and `model_metadata.csv`
- `tests/test_paper_artifacts.py`, `tests/test_generate_model_tables.py`,
  `tests/test_dataset_montage.py`, `tests/test_scale_scatter.py`
- ADR-0010, which governs them

**Still public, deliberately:**

- `src/croma/` — the library is the contribution
- `scripts/bench/` — the benchmarking pipeline is a stated deliverable of this
  repository, and `run_benchmarks.sh` moved here from `scripts/repro/` because its
  output destination is `output/metrics/`, not `paper/` (ADR-0006)
- `scripts/studies/` — the APD/nIPD source that `croma.downstream` is ported from
  (ADR-0011, issues #79–#84)
- `scripts/prep/`, `scripts/tools/`

## Consequences

- **ADR-0010 is local-only**, which is why the published `docs/adr/` skips from 0009
  to 0011. The number is not reused.
- **`scripts/studies/apd/loaders.py` and `bootstrap_uncertainty.py` import
  `paper_manifest`**, so those two studies cannot run from a fresh clone. No test
  imports them, so CI is unaffected. Porting them onto `croma.downstream` resolves it.
- Tracked files that *mention* local-only paths in prose are marked `(local-only)`
  rather than rewritten, so the pointer survives for whoever has the tree.
- The reproducibility claim narrows honestly: the repo publishes everything needed to
  compute the metrics and reproduce the benchmark numbers, not the LaTeX assembly.

## When this reverses

When the manuscript is public. At that point `paper/`, its artifacts and this tooling
can be tracked together, ADR-0003's original posture applies again as written, and the
freshness test becomes enforceable in CI because its inputs would finally be present.
Reversing is `git add` plus deleting the `.gitignore` block; nothing here is destructive.
