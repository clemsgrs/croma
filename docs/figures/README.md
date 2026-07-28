# Documentation figure sources

The site's figures are committed artifacts, not build-time products: the docs build runs
on a clean checkout that cannot see `paper/` or `output/`, both of which are git-ignored.

## Explanatory schematics

`src/*.tex` are standalone TikZ documents — the same sources the manuscript compiles —
tracked here so that every published figure has a public origin. They are self-contained;
each compiles on its own:

```bash
pdflatex -output-directory=<dir> docs/figures/src/croma_geometry.tex
```

## From source to site

```
src/<name>.tex  --pdflatex-->  <name>.pdf  --build_doc_figures.py-->  ../_static/figures/<name>-{light,dark}.svg
```

`scripts/tools/build_doc_figures.py` converts the rendered PDF to SVG and derives the dark
variant by transforming the *rendered figure*, not by recompiling it: neutral inks (the
near-black strokes and text, the near-white panel fills) have their lightness inverted,
while saturated colours — which carry meaning here, SO green, OS red, SS purple, OO orange,
anchor blue — keep their hue and are lifted just enough to hold contrast on a dark page.

Deriving the dark variant rather than compiling it is deliberate: it keeps the two variants
provably identical in geometry, and it means the figure set can be rebuilt on a machine with
no LaTeX toolchain.

One colour rule is worth knowing about, because it is not the obvious one. A colour is
inverted when it is a *neutral ink* **or** when it is simply very pale, whatever its hue —
nothing legible is drawn that pale on a white page, so a pale tint is always a background
wash and has to invert like one. Without that second clause the ridgelines' warm
fragile-region shade counted as a semantic colour, kept its lightness, and covered the left
half of every dark-mode ridgeline in a pale block.

## Results panels

`rank_pareto` and the three `distribution-<cohort>` ridgelines have no `.tex` source: the
same script draws them with the tracked benchmark plot library and then runs them through
the identical conversion, so a plotted figure and a schematic are themed by one code path.

```
results/cross_benchmark.csv  --plotting-->  <name>.pdf  --build_doc_figures.py-->  ../_static/figures/<name>-{light,dark}.svg
```

The Pareto panel is drawn from the committed `results/` export, so the encoders it rings are
the ones the table's `on_frontier` column marks. The ridgelines need every sample's CRoMa
rather than the 200-bin summary the export carries, so they read the run under `output/`
directly — which means, like the schematics, they can only be rebuilt where that tree exists.

They carry no TCGA-exposure daggers, unlike the manuscript's versions. The exposure flags
live in `scripts/repro/model_metadata.csv`, which is git-ignored, and a public figure must
not depend on a private file; the results pages carry that caveat in prose instead.

## Figures whose generator is not here

`dataset_cardinality` and `dataset_montage` come from the paper tooling under
`scripts/repro/`, which is local-only while the manuscript is unpublished (ADR-0012). They
are published as artifacts whose generator is not yet public; that gap closes when ADR-0012
reverses.
