.. _tcga-4x4:

TCGA-4×4
========

5,760 tiles spanning four cancer types — breast invasive carcinoma, colon adenocarcinoma,
and lung adeno- and squamous cell carcinoma — contributed by four TCGA tissue source sites
(Asterand, Christiana Healthcare, Roswell Park, University of Pittsburgh) and scored at
``k`` = :results-value:`k(tcga-4x4)`.

**Read this cohort with its pretraining overlap in mind.** TCGA is the most widely used
pretraining corpus in computational pathology, and many of these encoders have seen it. A
strong score can reflect an in-distribution advantage rather than robustness, and this page
cannot tell the two apart — the `paper <https://arxiv.org/abs/2607.25497>`_ quantifies the
overlap encoder by encoder. ``Midnight-12k``, which tops the cohort by a wide margin, is
pretrained on TCGA and on nothing else.

.. results-table:: tcga-4x4
   :caption: TCGA-4×4, sorted by median ``CRoMa``. Columns are explained under
             :ref:`result-columns`; † marks the natural-image control
             (:ref:`the-control`); row tint marks pretraining overlap (legend below).

.. _exposure-legend:

The row tint is that overlap, made visible, in three states rather than two because a
missing disclosure is not a clean bill:

- **Orange — TCGA-exposed.** TCGA appears in the encoder's disclosed pretraining corpus
  or institutional provenance. Discount an advantage here.
- **Grey — undisclosed.** A proprietary corpus that cannot be verified TCGA-free. Not an
  accusation; an absence of evidence.
- **Untinted — disclosed-clean.** A disclosed corpus without TCGA.

The states are derived from the tracked model registry's domain tags and published in
``results/cross_benchmark.csv``, so the tint here and the shading on the
:doc:`aggregate table <index>` come from one committed fact per model.

Only :results-value:`below_zero(tcga-4x4)` encoder falls below zero, and support is
near-total — every model sits at :results-value:`support_min(tcga-4x4)` or above, so ``RI``
and ``MaRI`` rest on essentially every tile. In the :ref:`explorer <explorer>`, every
encoder here resolves into a single tight mode close to zero, strong and weak alike; on
:ref:`Camelyon <camelyon>` the same panel spreads across most of the scale. Two cohorts,
one roster, very different separability — the argument for reporting more than one.

The two rankings, on this cohort alone:

.. themed-figure:: /_static/figures/pareto_tcga-4x4
   :alt: Median CRoMa against lower-tail mean on TCGA-4×4, one point per encoder,
         frontier ringed.

   Median ``CRoMa`` against tail severity LTM₁₀ on TCGA-4×4. Better is up and to the
   right; ringed points are undominated on both axes and named in bold. The natural-image
   control is excluded, and pretraining exposure is not marked — the caveat above applies
   to every point.
