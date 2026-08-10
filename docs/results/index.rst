.. _results:

Results
=======

:results-value:`roster()` encoders — :results-value:`ranked()` pathology foundation models
and one natural-image control — scored on three tile cohorts from the
`PathoROB <https://arxiv.org/abs/2507.17845>`_ study. Every number on this page, prose
included, is read at build time from :ref:`committed CSVs <results-provenance>` written by a
tracked exporter, never transcribed. The method is described in `Beyond counts: A
distributional robustness margin for pathology foundation models
<https://arxiv.org/abs/2607.25497>`_.

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

Each cohort has its own page: the full column set sorted by median ``CRoMa``, the cohort's
own median-versus-tail Pareto panel, and what is specific to reading it. Columns are
explained under :ref:`result-columns`; † marks the natural-image control
(:ref:`the-control`). Every encoder is evaluated at the cohort's shared operating point —
the cohort median of the per-model biological ``k*`` — with ``tau`` resolved per model
(see :ref:`results-provenance`).

.. toctree::
   :maxdepth: 1

   camelyon
   tcga-4x4
   tolkach-esca

:doc:`Camelyon <camelyon>` is scored entirely outside TCGA and is the most discriminating
of the three; :doc:`TCGA-4×4 <tcga-4x4>` must be read with pretraining overlap in mind;
:doc:`Tolkach-ESCA <tolkach-esca>` is the mildest, and the one where the count-based
indices stop separating models.

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

.. _results-provenance:

Where the numbers come from
---------------------------

Every number on this page is read at build time from
`results/ <https://github.com/clemsgrs/croma/tree/main/results>`_ — a small set of CSVs
committed to the repository. Nothing is transcribed, and nothing is fetched. The benchmark
runs themselves live under ``output/``, which is git-ignored and regenerable; the site
builds from a clean checkout and cannot see it. So a published number has to become a
committed artifact first, and
`scripts/tools/export_results.py <https://github.com/clemsgrs/croma/blob/main/scripts/tools/export_results.py>`_
is the only thing that writes one. The decision is recorded in
`ADR-0016 <https://github.com/clemsgrs/croma/blob/main/docs/adr/0016-results-is-a-committed-publication-artifact.md>`_.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Contents
   * - ``results/<cohort>.csv``
     - One row per encoder, holding exactly the columns the cohort tables render.
   * - ``results/cross_benchmark.csv``
     - The two pathology-only mean ranks and their average, the pathology frontier flag,
       and the median ``CRoMa`` and ``LTM₁₀`` behind them on each of the three cohorts.
       The DINOv2-B control keeps its measurements but has blank rank fields.
   * - ``results/distributions.json``
     - 200-bin histograms of per-sample ``CRoMa``, per encoder and cohort. What the
       explorer reads.
   * - ``results/PROVENANCE.json``
     - Protocol, per-cohort ``k``, ``tau`` policy, roster size, ``croma`` version, the
       source run for each cohort, and a sha256 of every file above.

The operating point
~~~~~~~~~~~~~~~~~~~

All three cohorts are reported under the **median-k** protocol: one shared ``k`` per cohort,
the cohort median of the per-model biological ``k*``. A single operating point is what makes
a rank across encoders meaningful — comparing a model evaluated at ``k = 5`` against one at
``k = 91`` compares two different questions.

``tau`` is never pinned. Each encoder gets the median typed-neighbour distance of its own
embedding at that ``k``, which is the only setting under which ``MaRI`` is comparable across
models (see :ref:`choosing-tau`). ``CRoMa`` is reported at its headline averaging radius,
``m = 5``, and LTM₁₀ at ``α = 0.10``.

The expanded panel
~~~~~~~~~~~~~~~~~~

The panel's two robustness-targeted fine-tunes (Mascaret, of Midnight-12k; Phaet, of
Phikon-v2) and the three-member RudolfV 2 teacher/student family were extracted under the
contract below, read from the completion sidecars. The full audit — runtimes, dimensions,
norm checks, digests — is committed in `docs/extraction-records/issue-130.md
<https://github.com/clemsgrs/croma/blob/main/docs/extraction-records/issue-130.md>`_.

.. list-table::
   :header-rows: 1
   :widths: 17 33 10 40

   * - Encoder
     - Checkpoint revision
     - Batch
     - Pooling
   * - Mascaret
     - ``e95e7ea15e039e78d74def101415e19d9a67ba80``
     - 32
     - ``checkpoint-native:model.encode``
   * - Phaet
     - ``e0ce6e0ee248470bd8604823e412ca64048a2495``
     - 64
     - ``checkpoint-native:model.encode``
   * - RudolfV 2
     - ``482d9519c6a10fc22fbe5bcd6a87d5daf056643c``
     - 32
     - ``concatenate-cls-and-mean-patches``
   * - RudolfV 2-B
     - ``b2cb55c8fff8aaaf9cc16fda6d09bfb21dfc6db8``
     - 32
     - ``concatenate-cls-and-mean-patches``
   * - RudolfV 2-S
     - ``76abacd512a98c72a6db6192af9fc98313c3bd78``
     - 64
     - ``concatenate-cls-and-mean-patches``

All five used FP32 inference and FP32 ``.npy`` storage. Mascaret and Phaet use a 224 px
resize and center crop with the checkpoint's ``pixel_mean``/``pixel_std`` contract and
retain checkpoint-native output normalization. The RudolfV 2 family uses the released
224×224 bicubic, antialiased preprocessing; its pooling concatenates the CLS token with the
mean of 784 patch tokens after excluding eight register tokens, retaining native, non-unit
output norms.

One caveat matters when reading the tables: RudolfV 2's disclosed Charité/LMU institutional
corpus creates a possible institutional/source-domain overlap with the CHA component of
Tolkach-ESCA. Exact patient or slide overlap is unknown, so this does not establish
leakage.

Public cohort boundary
~~~~~~~~~~~~~~~~~~~~~~

The committed web export deliberately contains three cohorts: Camelyon, TCGA-4×4 and
Tolkach-ESCA. TCGA-2×2 was recomputed with the same 26-model roster and is available in the
local metric tree and manuscript supplement, but is not a fourth public cohort or committed
CSV. Prostate-shift and the whole-slide panels are outside this expansion.

Freshness
~~~~~~~~~

Re-running a benchmark does not update this site. Someone has to run the exporter and commit
the diff — deliberately, because that diff is the review surface for changing a public claim.

A test guards the gap: it regenerates ``results/`` from ``output/`` and fails on any
difference, so a run that was never republished is caught rather than silently leaving stale
numbers here. It skips where ``output/`` is absent, which is every machine but the one
holding the runs.

The manuscript remains a local-only build tree
(`ADR-0012 <https://github.com/clemsgrs/croma/blob/main/docs/adr/0012-paper-tooling-stays-local.md>`_),
but its tables, macros, captions and guarded prose are regenerated from the same live runs by
``scripts/repro/build_paper.py``. The local ``tests/test_paper_artifacts.py`` freshness gate
compares generator output with the ignored manuscript tree; it cannot run in a clean CI
checkout, so it is an explicit pre-publication gate rather than a hosted guarantee.
