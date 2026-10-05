# Benchmark selection is flagged, not adjusted

When an encoder's authors disclose that PathoROB RI on the cohorts this repository ranks
them on was used during the encoder's development, for example in checkpoint selection,
its rows are shaded yellow in every results table on the aggregate and cohort pages,
including the nIPD tables. The legend sits beneath each table that shows the tint. The
ranks are not adjusted.

The fact lives in `model_metadata.csv` as `benchmark_selection`, naming what was used
(`pathorob-ri` for Mettle, whose authors used PathoROB RI over these cohorts as one of
several criteria to choose their released checkpoint). The exporter publishes it as the
boolean `benchmark_selected` beside `tcga_exposed` in `cross_benchmark.csv`. The README
cannot shade a Markdown row, so it marks the encoder ‡ and prints the legend.

## Why a flag and not an adjustment

How much developing against the benchmark flatters a rank cannot be measured from the
outside: it depends on how many candidates were compared, how correlated the selection
metric is with ours, and what else drove the choice. Any correction would be a guess
presented as a number. A flag states what was disclosed and leaves the discount to the
reader.

## Why a row tint

It is the same kind of caveat as pretraining overlap, which the site already shows as an
orange row tint (ADR-0005). Shading both keeps one visual vocabulary for "discount this
row": orange for data overlap with the cohort's source, yellow for use of the benchmark
during development. A row carrying both is split between the two colours. The dagger
keeps its one meaning on the site, the natural-image control. Every shaded row also gets a
visually-hidden label, so the distinction survives screen readers, print and copy-paste.

## Consequences

- An empty `benchmark_selection` means *not disclosed*, not an audited absence. We flag
  what authors tell us; we do not infer selection.
- A disclosed selection never removes an encoder from the panel, its ranks or its
  frontier membership.
- The legend is worded for use of PathoROB RI. A disclosure of a different kind needs its
  own wording before it is published.
- The explorers and the Pareto panels do not shade; they carry no row to shade.
