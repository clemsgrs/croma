.. _camelyon:

Camelyon
========

20,400 breast lymph-node tiles, labelled tumour or normal, contributed by two medical
centers (RUMC and UMCU) and scored at ``k`` = :results-value:`k(camelyon)`. Scored entirely
outside TCGA, so no encoder holds an in-distribution advantage from its pretraining corpus
— the cleanest of the three cohorts to read, and the most discriminating:
:results-value:`below_zero(camelyon)` pathology encoders score below zero, meaning their
typical neighbourhood is closer to a different-biology tile from the *same* center than to
a same-biology tile from another.

.. results-table:: camelyon
   :caption: Camelyon, sorted by median ``CRoMa``. Columns are explained under
             :ref:`result-columns`; † marks the natural-image control
             (:ref:`the-control`).

Read the support column carefully here. Two biological classes across two centers is a
sparse neighbourhood: no encoder's support fraction clears
:results-value:`support_max(camelyon)`, and the floor is
:results-value:`support_min(camelyon)`. A high ``RI`` over that little evidence is not the
same claim as one over :ref:`TCGA-4×4 <tcga-4x4>`'s near-total support — the same two
indices, resting on very different amounts of evidence.

The two rankings, on this cohort alone:

.. themed-figure:: /_static/figures/pareto_camelyon
   :alt: Median CRoMa against lower-tail mean on Camelyon, one point per encoder,
         frontier ringed.

   Median ``CRoMa`` against tail severity LTM₁₀ on Camelyon. Better is up and to the
   right; ringed points are undominated on both axes and named in bold. The natural-image
   control is excluded — the frontier is a pathology-only claim.

The shape says more than the median. ``Virchow2`` and ``CONCH`` sit within
:results-value:`gap(camelyon, croma, Virchow2, CONCH)` of each other on median ``CRoMa`` —
indistinguishable on that column alone — while ``CONCH`` carries
:results-value:`ratio(camelyon, croma_f0, CONCH, Virchow2)` the confounder-dominant mass
and :results-value:`ratio(camelyon, croma_ltm10, CONCH, Virchow2)` the tail severity.
Overlay the two in the :ref:`explorer <explorer>` to see it.
