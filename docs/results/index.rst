.. _results:

Results
=======

:results-value:`roster()` encoders — :results-value:`ranked()` pathology foundation models
and one natural-image control — scored on three tile cohorts from the
`PathoROB <https://arxiv.org/abs/2507.17845>`_ study, plus a separate
:results-value:`models(pcabiop)`-encoder slide-level cohort, :doc:`PCaBiop <pcabiop>`,
published on its own page. Every number on this page is read at build time from the
`committed CSVs <https://github.com/clemsgrs/croma/tree/main/results>`_, never
transcribed. The method is described in `A distributional robustness margin
for pathology foundation models <https://arxiv.org/abs/2607.25497>`_.

.. _aggregate-table:

Two rankings, one frontier
--------------------------

.. aggregate-table::

**Bold** marks the Pareto frontier: the encoders no other pathology encoder beats on *both*
rankings at once. † marks the unranked natural-image control (see :ref:`the-control`).
Orange rows mark encoders whose disclosed pretraining overlaps TCGA — one of the three
cohorts behind these ranks (:ref:`legend <exposure-legend>`).

The ``CRoMa`` and tail ranks are each encoder's mean rank across the three cohorts — by
median ``CRoMa`` and by tail severity LTM₁₀ respectively — and the mean rank averages the
two. It is a reading order, not a score: a model can hold a strong median margin and still
be brittle on a subgroup, so the two ranks stay in view beside their average.
``Midnight-12k`` and ``H-optimus-1`` show the two ways that plays out — the former ranks
far better by median margin than by tail severity, the latter the opposite.

The claim the table makes is the frontier — a *set*, not an order. Anything on it is a
defensible choice; which one you want depends on whether you care more about the typical
sample or the worst tenth of them.

.. raw:: html

   <div class="croma-pareto" data-kind="rank">
     <noscript>The Pareto overview needs JavaScript; the same ranks are in the table
     above.</noscript>
   </div>

The same two rankings as axes. Better is up and to the right; the ringed, named points
are undominated, and the shaded region below-left of the staircase is dominated on both
axes. Hover or tab to any point to name it with its two mean ranks.

.. _explorer:

The distribution explorer
-------------------------

The per-sample ``CRoMa`` distribution is the object every number above is read from, so it
is shown whole. The list is every encoder on the cohort, in the tables' order; the
highlighted row is drawn in full beneath it. Click a row to move the detail, drag across
the detail curve to count the samples in any range, and pick a second encoder under
*Compare with* to overlay its shape on the same axes — the readout then counts both.

.. raw:: html

   <div id="croma-explorer" class="croma-explorer" data-panel="tile">
     <noscript>The distribution explorer needs JavaScript.</noscript>
   </div>

The curves are the per-sample ``CRoMa`` at the headline radius ``m = 5`` — the same
committed 200-bin export the tables read — smoothed for display with the manuscript
figures' kernel (Gaussian, at Scott's bandwidth). The drag readout counts the raw bins,
never the smoothed curve.

The cohorts
-----------

Each cohort has its own page: the full column set sorted by median ``CRoMa``, the cohort's
own median-versus-tail Pareto panel and pinned distribution explorer, and what is specific
to reading it. Columns are
explained under :ref:`result-columns`; † marks the natural-image control
(:ref:`the-control`). Every encoder is evaluated at the cohort's shared operating point —
the cohort median of the per-model biological ``k*`` — with ``tau`` resolved per model
(see :ref:`choosing-tau`).

.. toctree::
   :maxdepth: 1

   camelyon
   tcga-4x4
   tolkach-esca

:doc:`Camelyon <camelyon>` is scored entirely outside TCGA and is the most discriminating
of the three; :doc:`TCGA-4×4 <tcga-4x4>` must be read with pretraining overlap in mind;
:doc:`Tolkach-ESCA <tolkach-esca>` is the mildest, and the one where the count-based
indices stop separating models.

The slide-level cohort
----------------------

:doc:`PCaBiop <pcabiop>` evaluates five *whole-slide* encoders — one embedding per slide —
on 1,000 PANDA prostate biopsies. It is a different roster on a different evaluation unit,
so it has its own page, its own distribution explorer, and no part in the aggregate ranks
above (see :ref:`cohort-caveats`).

.. toctree::
   :maxdepth: 1

   pcabiop

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

.. _the-control:

The natural-image control
-------------------------

``DINOv2-B``, the natural-image control, is pretrained on natural images and has never seen
a whole-slide image. Its measurements are shown with the panel, but it is excluded from the
pathology ranks and frontier because it is a calibration floor, not a competitor. Its
positive margin is not evidence that a natural-image model beats pathology encoders: it has
the *lowest* biological retrieval accuracy in the panel on every cohort, and ``CRoMa``
compares two neighbour distances, so a representation with weak structure of either kind
can score positively simply by having no strong confounder structure either. That is
precisely what makes it useful — it calibrates what a positive margin is worth on a poor
representation.

.. _cohort-caveats:

Scope
-----

The tile roster is fixed across the three tile cohorts, and the ranks above are computed
on it alone. The slide-level cohort :doc:`PCaBiop <pcabiop>` is a different roster, so it
never shares a table, a rank, or an explorer dropdown with the tile panel. And ranks are
within a panel: they say which of these encoders is more robust on these cohorts, not how
any of them would behave on yours.

The operating point
-------------------

The three tile cohorts are reported under the **median-k** protocol: one shared ``k`` per cohort,
the cohort median of the per-model biological ``k*``. A single operating point is what makes
a rank across encoders meaningful — comparing a model evaluated at ``k = 5`` against one at
``k = 91`` compares two different questions.

``tau`` is never pinned. Each encoder gets the median typed-neighbour distance of its own
embedding at that ``k``, which is the only setting under which ``MaRI`` is comparable across
models (see :ref:`choosing-tau`). ``CRoMa`` is reported at its headline averaging radius,
``m = 5``, and LTM₁₀ at ``α = 0.10``.

The slide-level cohort is the exception: with five encoders a shared median ``k`` is
dominated by panel composition, so :doc:`PCaBiop <pcabiop>` reports **k\*** — each encoder
at its own kNN-optimal ``k`` — and says so on its page.
