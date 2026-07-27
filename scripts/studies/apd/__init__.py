"""APD (Average Performance Drop) experiment trio.

Entrypoints (run as scripts, e.g. ``python scripts/studies/apd/apd_experiment.py``):
  - ``apd_experiment.py``      -- compute APD on croma embeddings (PathoROB-faithful).
  - ``apd_croma_correlation.py`` -- correlate APD with the faithful CRoMa/RI/MaRI metrics.

Neither owns a metric. The probe sweep and both reductions come from ``croma.downstream``
(ADR-0011), which is the code a reader installs, so the paper's downstream numbers are
produced by it. What is left here is the dataset configuration, the two croma-authored
schedules, the model roster, the resume cache and the CSVs.

The CRoMa-vs-APD scatter that consumes these outputs lives in
``scripts/repro/figures/apd_figure.py`` (a paper-figure emitter); it reaches the
shared data loaders and per-dataset configuration in ``loaders.py`` here via a
``sys.path`` shim. That emitter is local-only while the manuscript is unpublished and is
absent from a clone (ADR-0012), as is the ``paper_manifest`` naming which run each
benchmark's metrics come from -- which is why ``loaders`` imports it inside the join that
needs it rather than at module scope: computing APD must not wait on the manuscript's
build, only correlating it against RI/MaRI/CRoMa does.
"""
