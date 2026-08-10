Provenance
==========

Every number on these pages is read at build time from
`results/ <https://github.com/clemsgrs/croma/tree/main/results>`_ — a small set of CSVs
committed to the repository. Nothing is transcribed, and nothing is fetched.

The benchmark runs live under ``output/``, which is git-ignored and regenerable; the site
builds from a clean checkout and cannot see it. So a published number has to become a
committed artifact first, and
`scripts/tools/export_results.py <https://github.com/clemsgrs/croma/blob/main/scripts/tools/export_results.py>`_
is the only thing that writes one. The decision is recorded in
`ADR-0016 <https://github.com/clemsgrs/croma/blob/main/docs/adr/0016-results-is-a-committed-publication-artifact.md>`_.

What is committed
-----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Contents
   * - ``results/<cohort>.csv``
     - One row per encoder, holding exactly the columns the cohort pages render.
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
-------------------

All three cohorts are reported under the **median-k** protocol: one shared ``k`` per cohort,
the cohort median of the per-model biological ``k*``. A single operating point is what makes
a rank across encoders meaningful — comparing a model evaluated at ``k = 5`` against one at
``k = 91`` compares two different questions.

``tau`` is never pinned. Each encoder gets the median typed-neighbour distance of its own
embedding at that ``k``, which is the only setting under which ``MaRI`` is comparable across
models (see :ref:`choosing-tau`). ``CRoMa`` is reported at its headline averaging radius,
``m = 5``, and LTM₁₀ at ``α = 0.10``.

The expanded panel
------------------

The panel's two robustness-targeted fine-tunes (Mascaret, of Midnight-12k; Phaet, of
Phikon-v2) and the three-member RudolfV 2 teacher/student family were added under a
recorded extraction contract — checkpoint revisions, preprocessing, pooling, batch sizes,
dimension and norm checks. The complete audit is committed in
`docs/extraction-records/issue-130.md
<https://github.com/clemsgrs/croma/blob/main/docs/extraction-records/issue-130.md>`_.

One provenance caveat from it matters when reading the tables: RudolfV 2's disclosed
Charité/LMU institutional corpus creates a possible institutional/source-domain overlap
with the CHA component of Tolkach-ESCA. Exact patient or slide overlap is unknown, so this
does not establish leakage.

Public cohort boundary
----------------------

The committed web export deliberately contains three cohorts: Camelyon, TCGA-4×4 and
Tolkach-ESCA. TCGA-2×2 was recomputed with the same 26-model roster and is available in the
local metric tree and manuscript supplement, but is not a fourth public results page or
committed CSV. Prostate-shift and the whole-slide panels are outside this expansion.

Freshness
---------

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
