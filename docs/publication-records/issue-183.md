# Issue 183 — nIPD static publication record

The manuscript-facing nIPD interpretation was corrected alongside the public API and
documentation: values near zero indicate **little or no net change over the confounding
range**, not stable performance. Positive and negative portions of the signed trajectory
can cancel, so near-zero area cannot prove pointwise stability.

The manuscript assembly itself remains excluded from this repository under ADR-0003. The
corresponding authorial correction replaces “Values near zero indicate stable performance”
with the interpretation above. This record is its repository review surface without
committing the ignored `paper/` assembly.

Published data are `results/nipd.json` and `results/nipd.csv`, produced from
`output/studies/apd/` by `scripts/tools/export_nipd.py` on 2026-08-25 with croma 1.0.0.
