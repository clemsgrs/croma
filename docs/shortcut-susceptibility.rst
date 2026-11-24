Shortcut susceptibility
=======================

``nIPD`` measures the loss in supervised probe performance as training confounding
increases. A more negative value means greater integrated loss. **Zero nIPD** means
mean accuracy stayed at or above baseline at every sampled confounding level.

Interactive evidence browser
----------------------------

Select a cohort, then an encoder, to open the sampled trajectory behind its nIPD value.
Pick a second encoder under *Compare with* to overlay its trajectory on the same scale.
``DINOv2-B`` is shown separately as a natural-image control where available; it is not
ranked with the pathology encoders.

Rows are ranked by the normalized change at ``V = 1``, the endpoint of the trajectory,
while nIPD summarizes losses across the whole confounding range. A short drop at the
final level can contribute less to the area than a sustained smaller decline.
Rows ending at or below -90% are marked ``≈ chance``. The chance reference line
appears when the vertical axis reaches -100%.

.. raw:: html

   <div class="croma-nipd-explorer" data-payload="nipd.json"
        aria-label="Explore cohort-specific nIPD evidence">
     <p>Loading the committed nIPD evidence…</p>
   </div>
   <noscript>
     <p>Static nIPD tables for every cohort and regime:
     <a href="results/camelyon.html">Camelyon</a>,
     <a href="results/tcga-4x4.html">TCGA-4×4</a>,
     <a href="results/tolkach-esca.html">Tolkach-ESCA</a>,
     <a href="results/pcabiop.html">PCaBiop</a>.</p>
   </noscript>

The **CRoMa and downstream susceptibility** scatter below the trajectory pairs each
encoder's median CRoMa at ``m=5`` with its nIPD. Hover or tab to any point to name it
with its two values. The fitted trend and the Spearman coefficient use ranked pathology
encoders only, so ``DINOv2-B`` is excluded from both. PCaBiop holds ``n=5`` encoders,
so its coefficient is descriptive and no trend is fitted.

How nIPD is computed
--------------------

A logistic probe predicts the biological class from frozen embeddings. Its training
composition moves from balanced to fully confounded while the test rows stay fixed, and
at each Cramér's-``V`` value nIPD compares mean balanced accuracy with the balanced
baseline. At each sampled level, gains are set to zero and losses are divided by the
baseline's margin over chance. We integrate the resulting values by linear interpolation
between sampled levels. **nIPD measures the share of above-chance performance that is lost**.
**ID is the primary mechanistic
endpoint**: its acquisition groups remain represented in training, isolating
susceptibility to the shortcut. **OOD also includes transfer effects** because its
acquisition groups were unseen during training. See the :ref:`API definition <nipd-api>`
for the equation and input contract.

Results
-------

Per-cohort tables, ranked by the normalized change at ``V = 1``, live on the cohort pages:
:ref:`Camelyon <camelyon>`, :ref:`TCGA-4×4 <tcga-4x4>`, :ref:`Tolkach-ESCA
<tolkach-esca>` and :ref:`PCaBiop <pcabiop>`. Download the exact float basis as
`JSON <nipd.json>`_ or the complete tabular view as `CSV <nipd.csv>`_.

``APD`` remains available as the PathoROB-faithful continuity reduction. It normalizes
by raw baseline accuracy; nIPD is the primary result because it normalizes by the margin
over chance and integrates over the observed Cramér's-V coordinates.
