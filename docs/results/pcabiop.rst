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
:results-value:`roster()`-model tile panel, so this cohort takes no part in the
:ref:`cross-cohort aggregate <aggregate-table>` and its ranks — the table below orders
these five encoders against each other, nothing more. There is also no natural-image control at slide level, so no
row carries the † mark.

.. raw:: html

   <p><a href="https://github.com/clemsgrs/croma/issues/new?template=slide-encoder-request.yml">Don't see a slide encoder? Request an evaluation</a></p>

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

For binary cancer detection, we tested every integer ``k`` from 1 to 100. For each
encoder we selected the value that maximised biological k-NN balanced accuracy and used
it for both the biological and center accuracies, RI, MaRI and support. The selected
values span :results-value:`k_range(pcabiop)` (see ``results/PROVENANCE.json``).
With five slide encoders, the shared median can depend on one or two models, so we
report each encoder at its own operating point. CRoMa, *F*\ (0) and LTM₁₀ are
independent of ``k``.

Shortcut susceptibility
-----------------------

Shortcut susceptibility for the five whole-slide encoders, in domain (ID) and out of
domain (OOD). Rows are ranked by ``Change at V = 1``, the normalized change at maximum confounding.
nIPD integrates losses across the full confounding range, so the endpoint can reveal
a severe loss confined to the final level. Rows ending at or below −0.900 are marked
``≈ chance``.

Bold marks the leading ranked encoder in each column where higher is better, so a
column that disagrees with the ranking shows it at a glance. Each caption reports
Spearman ρ, the rank correlation between the ``CRoMa`` and ``nIPD`` columns: how closely
the two order the encoders the same way.
:doc:`../shortcut-susceptibility` defines the measure and holds the interactive explorer.

.. nipd-table:: pcabiop id

.. nipd-table:: pcabiop ood

Interpret MOOZY’s probe results in light of its pretraining on PANDA slides. The
out-of-domain PAR evaluation uses a separate cohort.
