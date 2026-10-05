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
             (:ref:`the-control`), and row shading is explained beneath the table.

.. _exposure-legend:

The orange rows are that overlap, made visible — the same convention as the paper's
dagger. Discount an advantage there. For the proprietary corpora, an unshaded row reflects
the paper's description of the corpus, not an independent audit.

Only :results-value:`below_zero(tcga-4x4)` encoder falls below zero, and support is
near-total — every model sits at :results-value:`support_min(tcga-4x4)` or above, so ``RI``
and ``MaRI`` rest on essentially every tile. In the explorer below, every
encoder here resolves into a single tight mode close to zero, strong and weak alike; on
:ref:`Camelyon <camelyon>` the same panel spreads across most of the scale. Two cohorts,
one roster, very different separability — the argument for reporting more than one.

The two rankings, on this cohort alone:

.. raw:: html

   <div class="croma-pareto" data-cohort="tcga-4x4">
     <noscript>The Pareto panel needs JavaScript; the same numbers are in the table
     above.</noscript>
   </div>

Median ``CRoMa`` against tail severity LTM₁₀ on TCGA-4×4. Better is up and to the right;
ringed points are undominated on both axes and named, and the shaded region is dominated
on both. Hover or tab to any point to name it with its two values. The natural-image
control is excluded, and pretraining exposure is not marked — the caveat above applies to
every point.

The distribution explorer
-------------------------

The same explorer as the :ref:`aggregate page's <explorer>`, pinned to TCGA-4×4. Click a
row to move the detail, drag across the detail curve to count the samples in any range,
and pick a second encoder under *Compare with* to overlay its shape.

.. raw:: html

   <div class="croma-explorer" data-cohort="tcga-4x4">
     <noscript>The distribution explorer needs JavaScript.</noscript>
   </div>

Shortcut susceptibility
-----------------------

Shortcut susceptibility for every encoder on this cohort, in domain (ID) and out of
domain (OOD); the natural-image control sits last. Rows are ranked by ``Change at V = 1``,
the normalized change at maximum confounding, because ``nIPD`` averages over the whole
range: an early gain there can pay for a late collapse, so a curve ending at chance can
outrank one that never moved. Rows ending at or below -0.900 are marked ``≈ chance``,
where none of the above-chance margin survives.

Bold marks the leading ranked encoder in each column where higher is better, so a
column that disagrees with the ranking shows it at a glance. Each caption reports
Spearman ρ, the rank correlation between the ``CRoMa`` and ``nIPD`` columns: how closely
the two order the encoders the same way.
:doc:`../shortcut-susceptibility` defines the measure and holds the interactive explorer.

.. nipd-table:: tcga-4x4 id

.. nipd-table:: tcga-4x4 ood
