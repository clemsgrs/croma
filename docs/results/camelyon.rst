.. _camelyon:

Camelyon
========

Camelyon contains 20,400 breast lymph-node tiles labelled tumour or normal from two
medical centers (RUMC and UMCU). We evaluated the encoders at
``k`` = :results-value:`k(camelyon)`. Of the 27 pathology encoders, GPFM had documented
training exposure to Camelyon. Available disclosures ruled out exposure for 15 encoders
and left it unresolved for 11.

Camelyon spans the widest range of median CRoMa values among the three primary tile
cohorts. For :results-value:`below_zero(camelyon)` pathology encoders, the median was
negative: a typical tile was closer to tiles of the other class from its own center than
to tiles of the same class from the other center.

.. results-table:: camelyon
   :caption: Camelyon, sorted by median ``CRoMa``. Columns are explained under
             :ref:`result-columns`; † marks the natural-image control
             (:ref:`the-control`), and row shading is explained beneath the table.

Read the support column carefully here. Two biological classes across two centers is a
sparse neighbourhood: no encoder's support fraction clears
:results-value:`support_max(camelyon)`, and the floor is
:results-value:`support_min(camelyon)`. A high ``RI`` over that little evidence is not the
same claim as one over :ref:`TCGA-4×4 <tcga-4x4>`'s near-total support — the same two
indices, resting on very different amounts of evidence.

The two rankings, on this cohort alone:

.. raw:: html

   <div class="croma-pareto" data-cohort="camelyon">
     <noscript>The Pareto panel needs JavaScript; the same numbers are in the table
     above.</noscript>
   </div>

Median ``CRoMa`` against tail severity LTM₁₀ on Camelyon. Better is up and to the
right; ringed points are undominated on both axes and named, and the shaded region is
dominated on both. Hover or tab to any point to name it with its two values. The
natural-image control is excluded — the frontier is a pathology-only claim.

The distribution explorer
-------------------------

The same explorer as the :ref:`aggregate page's <explorer>`, pinned to Camelyon. Click a
row to move the detail, drag across the detail curve to count the samples in any range,
and pick a second encoder under *Compare with* to overlay its shape.

.. raw:: html

   <div class="croma-explorer" data-cohort="camelyon">
     <noscript>The distribution explorer needs JavaScript.</noscript>
   </div>

The shape says more than the median. ``Virchow2`` and ``CONCH`` sit within
:results-value:`gap(camelyon, croma, Virchow2, CONCH)` of each other on median ``CRoMa`` —
indistinguishable on that column alone — while ``CONCH`` carries
:results-value:`ratio(camelyon, croma_f0, CONCH, Virchow2)` the confounder-dominant mass
and :results-value:`ratio(camelyon, croma_ltm10, CONCH, Virchow2)` the tail severity.
Overlay the two above to see it.

Shortcut susceptibility
-----------------------

Shortcut susceptibility for every encoder on this cohort, in domain (ID) and out of
domain (OOD); the natural-image control sits last. Rows are ranked by ``Change at V = 1``, the normalized change at maximum confounding.
nIPD integrates losses across the full confounding range, so the endpoint can reveal
a severe loss confined to the final level. Rows ending at or below −0.900 are marked
``≈ chance``.

Bold marks the leading ranked encoder in each column where higher is better, so a
column that disagrees with the ranking shows it at a glance. Each caption reports
Spearman ρ, the rank correlation between the ``CRoMa`` and ``nIPD`` columns: how closely
the two order the encoders the same way.
:doc:`../shortcut-susceptibility` defines the measure and holds the interactive explorer.

.. nipd-table:: camelyon id

.. nipd-table:: camelyon ood
