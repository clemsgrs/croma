# The benchmark driver is compute-only; rendering is a separate step

The benchmark driver computes metrics (RI, MaRI, CRoMa, tau, kNN sweeps, tail
stats), caches, and writes JSON/CSV — and does **not** render figures. A separate
render step takes a completed run directory and emits its figures from the written
data. The paper-figure re-render path is a thin wrapper over that same render module.

## Why

The paper figures were already produced by a separate step that reads the driver's
JSON, so the driver's inline rendering was only a convenience preview that also
duplicated the plot-call sequence. Splitting them lets compute run headless, keeps
matplotlib out of the compute path, and gives one definition of "which plots to
render" with two callers instead of two copies.

## Consequences

- The workflow is two commands (compute, then render); README and the shell drivers
  reflect this.
- A characterization test pins the compute output so the split stays behavior-
  preserving.
