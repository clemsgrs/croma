.. _pcabiop:

PCaBiop (slide-level)
=====================

1,000 prostate biopsy slides sourced from `PANDA
<https://www.kaggle.com/c/prostate-cancer-grade-assessment>`_, labelled benign or cancer,
contributed by two data providers (Karolinska Institutet and Radboud UMC). Unlike the
three tile cohorts, the evaluation unit is a whole slide: each of the
:results-value:`models(pcabiop)` encoders is a *whole-slide* foundation model producing
one embedding per slide, and every number below is computed over those slide embeddings.

**This is a different panel.** The slide roster shares no encoder ranking with the
26-model tile panel, so this cohort takes no part in the :ref:`cross-cohort aggregate
<aggregate-table>` and its ranks — the table below orders these five encoders against
each other, nothing more. There is also no natural-image control at slide level, so no
row carries the † mark.

.. results-table:: pcabiop
   :caption: PCaBiop, sorted by median ``CRoMa``. Columns are explained under
             :ref:`result-columns`.

The confounder is unusually dominant here: every encoder's confounder *k*-NN accuracy is
at or near 1.0 — the data provider is essentially perfectly decodable from every slide
embedding — and :results-value:`below_zero(pcabiop)` of the five encoders hold a negative
median margin. Support spans :results-value:`support_min(pcabiop)` to
:results-value:`support_max(pcabiop)`, the widest spread of any published cohort, so the
count-based indices rest on very different amounts of evidence per row: ``Prov-GigaPath``'s
low ``RI`` is measured on nearly every slide, while ``PRISM``'s rests on a tenth of them.

The two rankings, on this cohort alone:

.. raw:: html

   <div class="croma-pareto" data-cohort="pcabiop">
     <noscript>The Pareto panel needs JavaScript; the same numbers are in the table
     above.</noscript>
   </div>

Median ``CRoMa`` against tail severity LTM₁₀. Better is up and to the right; ringed
points are undominated on both axes. Hover or tab to any point to name it with its two
values. With five encoders the panel is sparse by construction — read it as a picture of
the table, not as a frontier claim of tile-panel strength.

The distribution explorer
-------------------------

The same explorer as the :ref:`tile panel's <explorer>`, restricted to the slide roster.
Click a row to move the detail, drag across the detail curve to count the slides in
any range, and pick a second encoder under *Compare with* to overlay its shape.

.. raw:: html

   <div class="croma-explorer" data-panel="slide">
     <noscript>The distribution explorer needs JavaScript.</noscript>
   </div>

The shape is again where the medians stop telling the story. ``PRISM2`` and ``PRISM`` sit
:results-value:`gap(pcabiop, croma, PRISM2, PRISM)` apart on median ``CRoMa`` —
indistinguishable on that column — while ``PRISM`` carries
:results-value:`ratio(pcabiop, croma_f0, PRISM, PRISM2)` the confounder-dominant mass and
:results-value:`ratio(pcabiop, croma_ltm10, PRISM, PRISM2)` the tail severity. Overlay
the two above to see it.

The operating point
-------------------

This cohort is reported at **k\***, not the tile panel's shared ``median-k``: each encoder
is evaluated at its own kNN-optimal ``k`` (here spanning ``k`` =
:results-value:`k_range(pcabiop)`; the per-encoder values are recorded in
``results/PROVENANCE.json``). With only five encoders, a shared median is dominated by
panel composition — adding a single encoder to the panel moved the would-be shared ``k``
from 3 to 9 — so pinning one ``k`` would make every number hostage to who else happens to
be on the roster. The trade is stated rather than hidden: ``RI`` and ``MaRI`` are
protocol-dependent, so those two columns are not measured at one shared operating point
here, while ``CRoMa``, *F*\ (0) and LTM₁₀ are ``k``-free and unaffected.
