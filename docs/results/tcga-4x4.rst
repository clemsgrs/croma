TCGA-4×4
========

5,760 tiles spanning four cancer types — breast invasive carcinoma, colon adenocarcinoma,
and lung adeno- and squamous cell carcinoma — contributed by four TCGA tissue source sites
(Asterand, Christiana Healthcare, Roswell Park, University of Pittsburgh).

**Read this cohort with its pretraining overlap in mind.** TCGA is the most widely used
pretraining corpus in computational pathology, and many of the encoders here have seen it.
A strong score can reflect an in-distribution advantage rather than robustness, and this
page cannot tell the two apart — the `paper <https://arxiv.org/abs/2607.25497>`_ quantifies
the overlap encoder by encoder.
``Midnight-12k``, which tops the cohort by a wide margin, is pretrained on TCGA and on
nothing else.

Only one encoder falls below zero here. Support is the other thing to notice: at ``k = 71``
every model sits above 99%, so ``RI`` and ``MaRI`` rest on essentially every tile. That is
worth contrasting with :doc:`Camelyon <camelyon>`, where support never exceeds 73% for any
encoder in the same panel — the same two indices, resting on very different amounts of
evidence.

.. results-table:: tcga-4x4
   :caption: TCGA-4×4 — all 26 encoders, by median ``CRoMa``

Columns are explained under :ref:`result-columns`; † marks the natural-image control
(:ref:`the-control`). Every model is evaluated at the shared operating point ``k = 71``,
the cohort median of the per-model biological ``k*``, with ``tau`` resolved per model.

The shape behind the numbers
----------------------------

.. themed-figure:: /_static/figures/distribution-tcga-4x4
   :alt: Per-sample CRoMa density for each encoder, sorted by median, with the
         confounder-dominant region shaded.

   Per-sample ``CRoMa``, one density per encoder, ordered by median. The shaded half is
   ``CRoMa < 0``: the mass sitting there is *F*\ (0), and LTM₁₀ is the mean of its worst
   tenth.

Every encoder here resolves into a single tight mode close to zero, strong and weak alike.
Compare :doc:`Camelyon <camelyon>`, where the same panel spreads across the whole domain
and the lower half of the table sits visibly to the left of it. Two cohorts, one roster,
very different separability — which is the argument for reporting more than one.
