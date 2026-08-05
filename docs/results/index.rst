.. _results:

Results
=======

Twenty-one encoders — twenty pathology foundation models and one natural-image control —
scored on three tile cohorts from the `PathoROB <https://arxiv.org/abs/2507.17845>`_ study.
Every number on these pages is read from :doc:`committed CSVs <provenance>` written by a
tracked exporter, never transcribed. The method behind them is described in
`Beyond counts: A distributional robustness margin for pathology foundation models
<https://arxiv.org/abs/2607.25497>`_.

.. _aggregate-table:

Two rankings, and an order to read them in
------------------------------------------

.. aggregate-table::

**Bold** marks the Pareto frontier: the encoders no other encoder beats on *both* rankings
at once. † marks the natural-image control (see :ref:`the-control` below).

The ``CRoMa`` and tail rank columns are the mean, across the three cohorts, of that
encoder's rank within each cohort — by median ``CRoMa`` for the first, by tail severity
``LTM₁₀`` for the second. The mean rank averages those two, and the table is sorted by it.

That aggregate is a reading order, not a score. It sorts the rows; it does not stand in for
the two columns it averages, both of which stay in view at the precision it was computed
from — so it can be re-derived, and so the disagreements it smooths over stay visible in the
same row. ``Midnight-12k`` and ``UNI2-h`` are the case worth looking at: they tie exactly on
the mean rank, and they are not similar models. ``Midnight-12k`` is 3rd on median margin and
16th on the tail; ``UNI2-h`` is middling on the margin and comfortably better on the tail.
A model can hold a strong median margin and still be brittle on a subgroup, and no single
number will tell you that — which is why the mean sorts the rows and the two columns beside
it carry the finding.

So the claim this table makes is still the frontier — a *set*, not an order. Anything on it
is a defensible choice; which one you want depends on whether you care more about the
typical sample or the worst tenth of them.

.. themed-figure:: /_static/figures/rank_pareto
   :alt: Mean CRoMa rank against mean tail rank, one point per encoder, frontier ringed.

   The same two rankings as axes. Better is up and to the right; ringed points are
   undominated, and the shaded region below-left of the staircase is dominated on both
   axes.

.. _explorer:

Size the tail yourself
----------------------

*F*\ (0) and LTM₁₀ are two summaries of a shape. Pick a cohort and an encoder, then drag
across the histogram to count the samples in any range — how much mass sits below zero, how
far the worst of it goes, how much of it a threshold would catch.

.. raw:: html

   <div id="croma-explorer" class="croma-explorer">
     <noscript>
       The explorer needs JavaScript. The same distributions are drawn as static figures on
       each cohort page.
     </noscript>
   </div>

The histograms are 200 bins of the per-sample ``CRoMa`` at the headline radius, read from
the same committed export as the tables. Sample identifiers and tile thumbnails are
deliberately absent: identifiers cost megabytes per cohort for a lookup nobody can act on
without the cohort in hand, and thumbnails would mean redistributing three datasets under
three different licences.

.. _the-control:

Reading the control
-------------------

``DINOv2-B`` is pretrained on natural images and has never seen a whole-slide image. It is
ranked inline rather than banded off, because it lands mid-panel and hiding that would
misrepresent where the floor actually is — but it is a floor, not a competitor.

Its positive margin is not evidence that a natural-image model beats pathology encoders. It
has the *lowest* biological retrieval accuracy in the panel on every cohort, and ``CRoMa``
compares two neighbour distances: a representation with weak structure of either kind can
score positively simply by having no strong confounder structure either. That is precisely
what makes it useful — it calibrates what a positive margin is worth on a poor
representation.

.. _cohort-caveats:

What these numbers do and do not say
------------------------------------

- **TCGA-4×4 is drawn from TCGA**, and many of these encoders pretrain on it. An
  in-distribution advantage is not the same as robustness, so that cohort's ranks read
  differently from the other two — which are scored on centres outside TCGA. The
  `paper <https://arxiv.org/abs/2607.25497>`_ quantifies the overlap encoder by encoder;
  these pages do not.
- **The roster is fixed at 21 across all three cohorts.** Cohorts computed on a different
  roster — prostate (16 encoders) and a slide-level panel (4) — are deliberately not
  published here, because a table whose roster silently differs from the one beside it
  misleads more than it informs.
- **Ranks are within this panel.** They say which of these encoders is more robust on these
  cohorts, not how any of them would behave on yours.

.. _result-columns:

The columns on the cohort pages
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - bio bacc / conf bacc
     - Balanced accuracy of a *k*-NN classifier predicting the biological label and the
       centre. Diagnostics, not scores: a high confounder accuracy marks a representation
       that encodes the centre strongly, so its maximum is never bolded.
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
     - The mean of the lowest decile of the per-sample ``CRoMa`` distribution. How bad the
       worst tenth actually is.
   * - support
     - The fraction of samples that contribute to ``RI``/``MaRI`` at all. A high index over
       a thin support is not a strong result — see :ref:`undefined neighbourhoods
       <undefined-neighbourhoods>`.

Per-cohort detail
-----------------

Each cohort page carries the full column set and its ``CRoMa`` distribution.

.. toctree::
   :maxdepth: 1

   camelyon
   tcga-4x4
   tolkach-esca
   provenance
