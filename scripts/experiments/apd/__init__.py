"""APD (Average Performance Drop) experiment trio.

Entrypoints (run as scripts, e.g. ``python scripts/experiments/apd/apd_experiment.py``):
  - ``apd_experiment.py``      -- compute APD on croma embeddings (PathoROB-faithful).
  - ``apd_croma_correlation.py`` -- correlate APD with the faithful CRoMa/RI/MaRI metrics.
  - ``apd_figure.py``          -- CRoMa-vs-APD scatter for the APD-validation section.

Their shared data loaders and per-dataset configuration live in ``loaders.py``.
"""
