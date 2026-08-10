.. _results:

Results
=======

:results-value:`roster()` encoders — :results-value:`ranked()` pathology foundation models
and one natural-image control — scored on three tile cohorts from the
`PathoROB <https://arxiv.org/abs/2507.17845>`_ study. Every number on this page, prose
included, is read at build time from :doc:`committed CSVs <provenance>` written by a
tracked exporter, never transcribed. The method is described in `Beyond counts: A
distributional robustness margin for pathology foundation models
<https://arxiv.org/abs/2607.25497>`_.

.. _aggregate-table:

Two rankings, one frontier
--------------------------

.. aggregate-table::

**Bold** marks the Pareto frontier: the encoders no other pathology encoder beats on *both*
rankings at once. † marks the unranked natural-image control (see :ref:`the-control`).

The ``CRoMa`` and tail ranks are each encoder's mean rank across the three cohorts — by
median ``CRoMa`` and by tail severity LTM₁₀ respectively — and the mean rank averages the
two. It is a reading order, not a score: a model can hold a strong median margin and still
be brittle on a subgroup, so the two ranks stay in view beside their average.
``Midnight-12k`` and ``H-optimus-1`` show the two ways that plays out — the former ranks
far better by median margin than by tail severity, the latter the opposite.

The claim the table makes is the frontier — a *set*, not an order. Anything on it is a
defensible choice; which one you want depends on whether you care more about the typical
sample or the worst tenth of them.

.. themed-figure:: /_static/figures/rank_pareto
   :alt: Mean CRoMa rank against mean tail rank, one point per encoder, frontier ringed.

   The same two rankings as axes. Better is up and to the right; ringed points are
   undominated, and the shaded region below-left of the staircase is dominated on both
   axes.

.. _explorer:

The distribution explorer
-------------------------

The per-sample ``CRoMa`` distribution is the object every number above is read from, so it
is shown whole. The list is every encoder on the cohort, in the tables' order; the
highlighted row is drawn in full beneath it. Click a row to move the detail, drag across
the large histogram to count the samples in any range, and pick a second encoder under
*Compare with* to overlay its shape on the same axes — the readout then counts both.

.. raw:: html

   <div id="croma-explorer" class="croma-explorer">
     <noscript>The distribution explorer needs JavaScript.</noscript>
   </div>

The histograms are 200 bins of the per-sample ``CRoMa`` at the headline radius ``m = 5``,
read from the same committed export as the tables. Sample identifiers and tile thumbnails
are deliberately absent: identifiers cost megabytes per cohort for a lookup nobody can act
on without the cohort in hand, and thumbnails would mean redistributing three datasets
under three different licences.

The cohorts
-----------

Each table below carries one cohort's full column set, sorted by median ``CRoMa``. Columns
are explained under :ref:`result-columns`; † marks the natural-image control
(:ref:`the-control`). Every encoder is evaluated at the cohort's shared operating point —
the cohort median of the per-model biological ``k*`` — with ``tau`` resolved per model
(see :doc:`provenance`).

.. _camelyon:

Camelyon
~~~~~~~~

20,400 breast lymph-node tiles, labelled tumour or normal, contributed by two medical
centers (RUMC and UMCU) and scored at ``k`` = :results-value:`k(camelyon)`. Scored entirely
outside TCGA, so no encoder holds an in-distribution advantage from its pretraining corpus
— the cleanest of the three cohorts to read, and the most discriminating:
:results-value:`below_zero(camelyon)` pathology encoders score below zero, meaning their
typical neighbourhood is closer to a different-biology tile from the *same* center than to
a same-biology tile from another.

.. results-table:: camelyon
   :caption: Camelyon, sorted by median ``CRoMa``

Read the support column carefully here. Two biological classes across two centers is a
sparse neighbourhood: no encoder's support fraction clears
:results-value:`support_max(camelyon)`, and the floor is
:results-value:`support_min(camelyon)`. A high ``RI`` over that little evidence is not the
same claim as one over :ref:`TCGA-4×4 <tcga-4x4>`'s near-total support — the same two
indices, resting on very different amounts of evidence.

The shape says more than the median. ``Virchow2`` and ``CONCH`` sit within
:results-value:`gap(camelyon, croma, Virchow2, CONCH)` of each other on median ``CRoMa`` —
indistinguishable on that column alone — while ``CONCH`` carries
:results-value:`ratio(camelyon, croma_f0, CONCH, Virchow2)` the confounder-dominant mass
and :results-value:`ratio(camelyon, croma_ltm10, CONCH, Virchow2)` the tail severity.
Overlay the two in the :ref:`explorer <explorer>` to see it.

.. _tcga-4x4:

TCGA-4×4
~~~~~~~~

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
   :caption: TCGA-4×4, sorted by median ``CRoMa``

Only :results-value:`below_zero(tcga-4x4)` encoder falls below zero, and support is
near-total — every model sits at :results-value:`support_min(tcga-4x4)` or above, so ``RI``
and ``MaRI`` rest on essentially every tile. In the explorer, every encoder here resolves
into a single tight mode close to zero, strong and weak alike; on Camelyon the same panel
spreads across most of the scale. Two cohorts, one roster, very different separability —
the argument for reporting more than one.

.. _tolkach-esca:

Tolkach-ESCA
~~~~~~~~~~~~

9,000 tiles of oesophageal tissue across six classes — tumour, regression, adventitia,
muscularis propria, oesophageal and gastric mucosa — from three centers (UKK, WNS and CHA),
scored at ``k`` = :results-value:`k(tolkach-esca)`. The TCGA cohort of the original Tolkach
dataset is held out, following PathoROB, so like Camelyon this is scored outside TCGA.

.. results-table:: tolkach-esca
   :caption: Tolkach-ESCA, sorted by median ``CRoMa``

The mildest of the three cohorts — :results-value:`below_zero(tolkach-esca)` encoders fall
below zero — and the one where the count-based indices run out of room:
:results-value:`count_above(tolkach-esca, ri, 0.9)` of the :results-value:`ranked()` ranked
encoders score above ``0.90`` on ``RI``, so ``RI`` and ``MaRI`` have largely stopped
separating models here while ``CRoMa`` still spreads the panel. Several encoders are also
visibly bimodal in the explorer — one population of neighbourhoods comfortably
biology-dominant, another close to the line. A median reports where the middle of that
lands and says nothing about the split, which is the case tail reporting exists for.

.. _result-columns:

Reading the columns
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - bio bacc / conf bacc
     - Balanced accuracy of a *k*-NN classifier predicting the biological label and the
       center. Diagnostics, not scores: a high confounder accuracy marks a representation
       that encodes the center strongly, so its maximum is never bolded.
   * - ``RI`` / ``MaRI``
     - The pooled count-based and distance-weighted indices, in ``[0, 1]``, neutral at
       ``0.5``. See :doc:`../metrics`.
   * - Δ
     - ``MaRI − RI``. Informative in its sign — whether weighting by distance helps or
       hurts — rather than ordered by its size, so it is never bolded either.
   * - ``CRoMa``
     - The median signed margin at the headline radius, in ``(-1, 1)``, neutral at ``0``.
   * - *F*\ (0)
     - The fraction of samples with ``CRoMa <= 0``: confounder-dominant neighbourhoods,
       over the samples on which ``CRoMa`` is defined. Lower is better, so its bold is the
       minimum. See :ref:`confounder-dominant-fraction`.
   * - LTM₁₀
     - The lower-tail mean: the mean of the lowest decile of the per-sample ``CRoMa``
       distribution. How bad the worst tenth actually is.
   * - support
     - The support fraction: how many samples contribute to ``RI``/``MaRI`` at all. A high
       index over a thin support is not a strong result — see :ref:`undefined
       neighbourhoods <undefined-neighbourhoods>`.

.. _cohort-caveats:

What these numbers do and do not say
------------------------------------

.. _the-control:

The natural-image control
~~~~~~~~~~~~~~~~~~~~~~~~~

``DINOv2-B``, the natural-image control, is pretrained on natural images and has never seen
a whole-slide image. Its measurements are shown with the panel, but it is excluded from the
pathology ranks and frontier because it is a calibration floor, not a competitor. Its
positive margin is not evidence that a natural-image model beats pathology encoders: it has
the *lowest* biological retrieval accuracy in the panel on every cohort, and ``CRoMa``
compares two neighbour distances, so a representation with weak structure of either kind
can score positively simply by having no strong confounder structure either. That is
precisely what makes it useful — it calibrates what a positive margin is worth on a poor
representation.

Scope
~~~~~

The roster is fixed across all three cohorts; cohorts computed on a
different roster — a prostate panel and a slide-level panel — are deliberately not
published here, because a table whose roster silently differs from the one beside it
misleads more than it informs. And ranks are within this panel: they say which of these
encoders is more robust on these cohorts, not how any of them would behave on yours.

Where the numbers come from
---------------------------

.. toctree::
   :maxdepth: 1

   provenance
