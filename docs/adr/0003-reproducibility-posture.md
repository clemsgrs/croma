# Reproducibility posture: commit the generators, not the manuscript or the blobs

> **Publication transition recorded by
> [ADR-0012](0012-paper-tooling-stays-local.md).** The manuscript is public and the `.tex`
> and figure generators are committed, so this reproducibility posture applies in full.

The public repo contains everything needed to *reproduce* the paper — the library,
tests, and all scripts including `scripts/repro/` (the figure/table/value
generators) — plus `CONTEXT.md` and `docs/adr/`. It does **not** contain the paper
`.tex` source, the raw `data/`, or the `output/` blobs.

## Why

Reproducibility is a project goal, so the code that turns embeddings into every
number and figure in the paper must be public and runnable. But the manuscript
draft belongs on arXiv (not coupled to repo history), and the datasets/embeddings
are large and are released separately (HuggingFace). Committing generators without
the manuscript or the blobs gives full reproducibility at a lean public surface.

A reproducer clones the repo, `pip install croma[repro]`, pulls the released
datasets, and re-runs the scripts to regenerate `paper/` and `output/` locally.

## gitignore consequences

- **Now committed (previously ignored):** `docs/` (for `docs/adr/`) and the paper generators
  under `scripts/repro/`. `CONTEXT.md` was already trackable.
- **Stays ignored (local-only):** `paper/`, `data/`, `output/`, `resources/`,
  `tasks/`, `.tasks/`, and the dev scratch files (`WRAP-UP.html`, `feedback.html`,
  `pathorob-ri.png`, `TODO.md`). `AGENTS.md`/`CLAUDE.md` remain ignored unless we
  decide to publish them as contributor guidance.
