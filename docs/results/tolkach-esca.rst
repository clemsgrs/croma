.. _tolkach-esca:

Tolkach-ESCA
============

9,000 tiles of oesophageal tissue across six classes — tumour, regression, adventitia,
muscularis propria, oesophageal and gastric mucosa — from three centers (UKK, WNS and CHA),
scored at ``k`` = :results-value:`k(tolkach-esca)`. The TCGA cohort of the original Tolkach
dataset is held out, following PathoROB, so like :ref:`Camelyon <camelyon>` this is scored
outside TCGA.

.. results-table:: tolkach-esca
   :caption: Tolkach-ESCA, sorted by median ``CRoMa``. Columns are explained under
             :ref:`result-columns`; † marks the natural-image control
             (:ref:`the-control`).

One provenance caveat: the ``RudolfV-2`` family's disclosed Charité/LMU institutional
corpus creates a possible institutional/source-domain overlap with this cohort's CHA
center. Exact patient or slide overlap is unknown, so this does not establish leakage —
but read the family's scores here with it in mind.

The mildest of the three cohorts — :results-value:`below_zero(tolkach-esca)` encoders fall
below zero — and the one where the count-based indices run out of room:
:results-value:`count_above(tolkach-esca, ri, 0.9)` of the :results-value:`ranked()` ranked
encoders score above ``0.90`` on ``RI``, so ``RI`` and ``MaRI`` have largely stopped
separating models here while ``CRoMa`` still spreads the panel.

The two rankings, on this cohort alone:

.. raw:: html

   <div class="croma-pareto" data-cohort="tolkach-esca">
     <noscript>The Pareto panel needs JavaScript; the same numbers are in the table
     above.</noscript>
   </div>

Median ``CRoMa`` against tail severity LTM₁₀ on Tolkach-ESCA. Better is up and to the
right; ringed points are undominated on both axes and named, and the shaded region is
dominated on both. Hover or tab to any point to name it with its two values. The
natural-image control is excluded — the frontier is a pathology-only claim.

Several encoders are also visibly bimodal in the :ref:`explorer <explorer>` — one
population of neighbourhoods comfortably biology-dominant, another close to the line. A
median reports where the middle of that lands and says nothing about the split, which is
the case tail reporting exists for.
