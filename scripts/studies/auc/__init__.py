"""Threshold-free arm of the downstream study: nIPD from AUROC beside balanced accuracy.

nIPD is computed from balanced accuracy, which depends on a decision threshold. Under
shortcut training part of the measured degradation can be a pure shift of the decision
boundary -- the probe still ranks samples correctly, its threshold drifts -- and AUROC is
blind to such a shift. This study re-runs the APD study's probe sweep, scores every probe
with both, and asks whether the association with representation robustness survives.

Entrypoints (run as scripts):
  - ``probe_auc_cell.py`` -- one (cohort, encoder) cell: the APD sweep, observed.
  - ``run_sweep.sh``      -- every missing or stale cell, a few at a time; resumable.
  - ``auc_cells.py``      -- the cell list (the APD study's) and the staleness check.
  - ``report.py``         -- the reduction CSV and the correlations behind
    ``tab:nipd-auroc``.

It owns no protocol. The sweep is ``apd_experiment.compute``, and both reductions are
``croma.nipd`` on the APD study's Cramér's-V grid.
"""
