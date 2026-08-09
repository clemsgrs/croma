# Model metadata is a single machine-readable source that generates the model tables

Per-model facts (params, #WSIs, #tiles, regime, embedding dim, pretraining corpus,
TCGA exposure), checkpoint relationships (family, parent, variant role, shared training
run/corpus), institutional provenance domains, and plot identity (family tone and order)
live in one machine-readable metadata table at `scripts/bench/model_metadata.csv`. A generator
emits the **table bodies** of the model-summary and model-provenance sections from it, and the
same source feeds the CRoMa-vs-scale figure and shared plot style. In/out-of-distribution status
is derived from corpus or institutional provenance intersected with the benchmark domain rather
than hand-kept.

## Why

The metadata previously existed only as a hand-authored LaTeX table, so it could not
be plotted and drifted from any other use. Single-sourcing the numbers means editing
one place updates the tables and the figure together, and the scale figure never
disagrees with the table.

Structured variant roles are presentation-bearing: the table identifies robustness-targeted
fine-tunes and teacher/distilled-student relationships from `parent_model` and `variant_role`
rather than embedding those relationships in prose-only cells. A shared-corpus identifier makes
one family training corpus auditable across every variant row.

## Considered options

- Fully generate the tables including captions/footnotes — rejected: the elaborate
  per-model footnotes are where formatting fidelity breaks. Only the table **bodies**
  (the numbers, which drift) are generated; the `\caption`, footnote markers, and
  prose stay in a hand-maintained template wrapper.

## Consequences

- Some #WSIs are undisclosed by the model cards; those are filled from PathoROB and
  carry a superscript source marker so provenance stays honest and table/figure agree.
- Institutional exposure is conservative. A shared institution or source domain can be marked
  without asserting that any benchmark patient or slide appeared in pretraining.
- The registry belongs to the benchmark layer rather than `scripts/repro/`: runtime plot
  identity ships with `scripts/bench/`, while the source distribution continues to exclude all
  paper tooling (ADR-0012 and ADR-0017).
