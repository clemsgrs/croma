"""APD (Average Performance Drop) experiment trio.

Entrypoints (run as scripts, e.g. ``python scripts/studies/apd/apd_experiment.py``):
  - ``apd_experiment.py``      -- compute APD on croma embeddings (PathoROB-faithful).
  - ``apd_croma_correlation.py`` -- correlate APD with the faithful CRoMa/RI/MaRI metrics.

The CRoMa-vs-APD scatter that consumes these outputs lives in
``scripts/repro/figures/apd_figure.py`` (a paper-figure emitter); it reaches the
shared data loaders and per-dataset configuration in ``loaders.py`` here via a
``sys.path`` shim.
"""
