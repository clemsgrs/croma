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

## Figures whose generator is not here

`dataset_cardinality` and `dataset_montage` come from the paper tooling under
`scripts/repro/`, which is local-only while the manuscript is unpublished (ADR-0012). They
are published as artifacts whose generator is not yet public; that gap closes when ADR-0012
reverses.
