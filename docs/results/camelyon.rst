Camelyon
========

20,400 breast lymph-node tiles, labelled tumour or normal, contributed by two medical
centres (RUMC and UMCU). Scored entirely outside TCGA, so no encoder here holds an
in-distribution advantage from its pretraining corpus — which makes this the cleanest of
the three cohorts to read.

It is also the most discriminating of the three. Seven encoders score below zero, meaning
the typical neighbourhood is closer to a different-biology tile from the *same* centre than
to a same-biology tile from another.

.. results-table:: camelyon
   :caption: Camelyon — all 26 encoders, by median ``CRoMa``

Columns are explained under :ref:`result-columns`; † marks the natural-image control
(:ref:`the-control`). Every model is evaluated at the shared operating point ``k = 11``,
the cohort median of the per-model biological ``k*``, with ``tau`` resolved per model.

Read the support column carefully here. Two biological classes across two centres is a
sparse neighbourhood, and no encoder in this panel clears 73% — several sit near 10%. A high
``RI`` over that little evidence is not the same claim as a high ``RI`` over a full cohort,
which is why :doc:`TCGA-4×4 <tcga-4x4>` (support above 99% throughout) reads so differently.

The shape behind the numbers
----------------------------

.. themed-figure:: /_static/figures/distribution-camelyon
   :alt: Per-sample CRoMa density for each encoder, sorted by median, with the
         confounder-dominant region shaded.

   Per-sample ``CRoMa``, one density per encoder, ordered by median. The shaded half is
   ``CRoMa < 0``: the mass sitting there is *F*\ (0), and LTM₁₀ is the mean of its worst
   tenth.

This is what the two tail columns summarise, and why they are worth reading. ``Virchow2``
and ``CONCH`` sit within ``0.003`` of each other on median ``CRoMa`` — indistinguishable on
that column alone — while ``CONCH`` carries three quarters again as much
confounder-dominant mass and twice the tail severity. A single pooled score cannot separate
them; the shape does.
