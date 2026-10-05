# The shared k floats with the panel

Each tile cohort's shared operating point stays the lower median of the per-model `k*`
over the *current* roster. When an encoder joins the published panel, the shared `k` is
recomputed, and it may move. Every number read at it moves with it: the two kNN
accuracies, RI, MaRI and support. CRoMa, F(0) and LTM₁₀ do not depend on `k`, so neither do
the ranks.

Two rules make the drift visible rather than silent:

- Every cohort table's caption states the `k` and the roster size it was taken over
  ("`k` = 71, the median of the 27 encoders' best `k`"), read from `PROVENANCE.json` at
  build time.
- The CHANGELOG records every move ("Tolkach-ESCA from `k = 61` to `k = 71`") and names
  the change that caused it.

## Considered options

- **Freeze `k` per cohort between versioned protocol bumps.** Published RI and MaRI would
  stay stable as encoders join, and a new encoder would be scored at the frozen `k`. We
  rejected it because it makes the operating point a constant that has to be maintained
  and justified separately from the panel. A floating `k` remains, by definition, the
  panel median the paper describes.
- **Drop RI and MaRI from the site.** Rejected: RI is the metric PathoROB and model cards
  report, and the site's RI is the bridge to them.

## Consequences

- A published RI or MaRI is a statement about the panel at the time it was exported. A
  value quoted from the site can go stale when an encoder joins; the caption and the
  CHANGELOG are how a reader reconciles it.
- A results PR that adds an encoder can rewrite every row of a cohort's k-dependent
  columns. Review compares the new rows against the old under the old `k` before
  accepting the move.
- The paper fixes its own roster, so its `k` is fixed with it.
