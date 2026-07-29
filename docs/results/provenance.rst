Provenance
==========

Every number on these pages is read at build time from
`results/ <https://github.com/clemsgrs/croma/tree/main/results>`_ — a small set of CSVs
committed to the repository. Nothing is transcribed, and nothing is fetched.

That indirection exists for a reason. The benchmark runs live under ``output/``, which is
git-ignored and regenerable; the site builds from a clean checkout and cannot see it. So a
published number has to become a committed artifact first, and
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
     - The two mean ranks, the frontier flag, and the three per-cohort margins.
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

Staying honest
--------------

Re-running a benchmark does not update this site. Someone has to run the exporter and commit
the diff — deliberately, because that diff is the review surface for changing a public claim.

A test guards the gap: it regenerates ``results/`` from ``output/`` and fails on any
difference, so a run that was never republished is caught rather than silently leaving stale
numbers here. It skips where ``output/`` is absent, which is every machine but the one
holding the runs.

The manuscript's tables are assembled by a separate, currently local-only toolchain
(`ADR-0012 <https://github.com/clemsgrs/croma/blob/main/docs/adr/0012-paper-tooling-stays-local.md>`_).
The two paths render overlapping numbers and are not unified. They agree today; nothing here
enforces that they always will.
