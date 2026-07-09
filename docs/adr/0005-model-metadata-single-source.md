# Model metadata is a single machine-readable source that generates the model tables

Per-model facts (params, #WSIs, #tiles, regime, embedding dim, pretraining corpus,
TCGA-exposure) live in one machine-readable metadata table. A generator emits the
**table bodies** of the model-summary and model-provenance sections from it, and the
same source feeds the CRoMa-vs-scale figure. In/out-of-distribution status is derived
from corpus ∩ benchmark domain rather than hand-kept.

## Why

The metadata previously existed only as a hand-authored LaTeX table, so it could not
be plotted and drifted from any other use. Single-sourcing the numbers means editing
one place updates the tables and the figure together, and the scale figure never
disagrees with the table.

## Considered options

- Fully generate the tables including captions/footnotes — rejected: the elaborate
  per-model footnotes are where formatting fidelity breaks. Only the table **bodies**
  (the numbers, which drift) are generated; the `\caption`, footnote markers, and
  prose stay in a hand-maintained template wrapper.

## Consequences

- Some #WSIs are undisclosed by the model cards; those are filled from PathoROB and
  carry a superscript source marker so provenance stays honest and table/figure agree.
