# Reproduction scripts are grouped by output destination, not by code coupling

The `scripts/` reproduction layer is organised into five role clusters — `bench/`
(engine), `prep/` (data prep), `repro/` (paper artifacts, with `repro/figures/`
holding **every** paper-figure emitter), `studies/` (claim-backing experiments +
shared helpers), and `tools/` (dev/exploratory). The placement rule is **output
destination**: a script that writes into the paper figures tree lives in
`repro/figures/` regardless of what it imports.

## Why

"All paper figures in one place" is the most useful axis for someone assembling or
reproducing the paper — it beats grouping by which shared helper a script imports.

## Considered options

- Group by coupling family (keep each script next to the helper it imports) — the
  more obvious choice, and rejected. A future reader will notice that a shared helper
  (e.g. the neighborhood-analysis loader) sits in `studies/` while several of its
  importers sit in `repro/figures/`, and wonder why. The answer: helpers stay with
  their conceptual family and are reached across directories via the existing
  sys.path-shim idiom; the figure scripts were placed by where their output goes, not
  by what they import. Cross-directory imports are acceptable because the shim
  pattern already exists project-wide (invocation stays path-based; no
  package-ification — see ADR-0002/0003 for why `scripts/` is repo-only tooling).
