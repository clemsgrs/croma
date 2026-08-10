Tolkach-ESCA
============

9,000 tiles of oesophageal tissue across six classes — tumour, regression, adventitia,
muscularis propria, oesophageal and gastric mucosa — from three centres (UKK, WNS and
CHA). The TCGA cohort of the original Tolkach dataset is held out, following PathoROB, so
like Camelyon this is scored outside TCGA.

This is the mildest of the three cohorts: no encoder's median falls below zero, and the top
of the panel keeps confounder-dominant mass under 6%. Margins are correspondingly wide, and
every median sits to the right of the neutral line.

It is also where the count-based indices run out of room. Sixteen of the twenty-six encoders
score above ``0.90`` on ``RI``, and the whole panel spans a third of the scale — so ``RI``
and ``MaRI`` have largely stopped separating models here, while ``CRoMa`` still spreads them
across a factor of five. That is the case these metrics were built for.

.. results-table:: tolkach-esca
   :caption: Tolkach-ESCA — all 26 encoders, by median ``CRoMa``

Columns are explained under :ref:`result-columns`; † marks the natural-image control
(:ref:`the-control`). Every model is evaluated at the shared operating point ``k = 61``,
the cohort median of the per-model biological ``k*``, with ``tau`` resolved per model.

The shape behind the numbers
----------------------------

.. themed-figure:: /_static/figures/distribution-tolkach-esca
   :alt: Per-sample CRoMa density for each encoder, sorted by median, with the
         confounder-dominant region shaded.

   Per-sample ``CRoMa``, one density per encoder, ordered by median. The shaded half is
   ``CRoMa < 0``: the mass sitting there is *F*\ (0), and LTM₁₀ is the mean of its worst
   tenth.

Several encoders here are visibly bimodal — two populations of neighbourhood, one
comfortably biology-dominant and one close to the line. A median reports where the middle
of that lands and says nothing about the split, which is the case tail reporting exists
for.
