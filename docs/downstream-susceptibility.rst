Downstream shortcut susceptibility
==================================

``nIPD`` is a signed downstream-susceptibility measure. **Negative nIPD means net
degradation** as training-set confounding increases, **near zero means little or no net
change over the confounding range**, and **positive nIPD means net improvement**. A
near-zero signed area can include offsetting changes, so it is not proof of stable
performance at every point.

Interactive evidence browser
----------------------------

Select a cohort first, then inspect the cohort's pathology encoders on one signed nIPD
axis. The overview runs from more degradation toward **less degradation**: negative is net
degradation, zero is no net change, and positive is net improvement. Selecting an encoder
opens the real sampled trajectory that supplies its scalar. ``DINOv2-B`` is shown
separately as a natural-image control where it is available; it is not ranked with the
pathology encoders.

ID is the default mechanistic view. OOD also reflects transfer effects from acquisition
groups unseen during training. The trajectory scale stays fixed while switching models
within a cohort and regime. Its patterned signed lobes make cancellation visible, and the
sampled points expose the paired-repeat 95% t-intervals used to describe uncertainty.
Optionally overlay one second encoder: both trajectories and intervals stay visible on the
same fixed scale, while patterned signed-area shading and the active readouts belong to one
model at a time. **Swap active model** transfers that emphasis without clearing the pair.

The separate **CRoMa and downstream susceptibility** scatter below the trajectory uses
median CRoMa at ``m=5`` and nIPD from the active cohort and regime. Selecting a point opens
that encoder's trajectory, and selecting an encoder above highlights the same point. The
fitted trend and Spearman coefficient use ranked pathology encoders only. ``DINOv2-B`` is
shown as a distinct natural-image reference where available, but is excluded from the
fitted trend and Spearman correlation.

.. raw:: html

   <div class="croma-nipd-explorer" data-payload="nipd.json"
        aria-label="Explore cohort-specific nIPD evidence">
     <p>Loading the committed nIPD evidence…</p>
   </div>
   <noscript>
     <p><strong>Interactive explorer unavailable:</strong> JavaScript is disabled.
     The static tables below are the complete no-JavaScript equivalent.</p>
   </noscript>

Higher CRoMa is **associated with nIPD** in these within-cohort model panels. This does
not establish that CRoMa causes downstream performance. Representation metrics and
downstream probes can use different evaluation samples, so this is a model-level
comparison, not sample-level pairing. PCaBiop contains ``n=5`` encoders; its Spearman
coefficient is descriptive, and the fitted trend is omitted to avoid giving it the same
inferential weight as the 25-model tile panels.

The static tables below are the complete no-JavaScript equivalent and use the same
committed payload as the explorer. They include the exact median CRoMa and nIPD source
values and ranked-panel Spearman coefficient for every cohort and regime.

The confounder-biased probe sweep trains a logistic probe to predict the biological class
from frozen embeddings. Its training composition moves from balanced to fully confounded
while the test rows stay fixed. At each Cramér's-``V`` value, nIPD compares mean balanced
accuracy with the balanced baseline.

The denominator is **baseline skill: baseline balanced accuracy minus chance**. It
therefore measures change relative to the above-chance performance available to lose,
rather than normalizing by accuracy that includes the irreducible chance floor. See the
:ref:`API definition <nipd-api>` for the equation and input contract.

ID and OOD answer related but different questions. **ID is the primary mechanistic
endpoint**: acquisition groups remain represented, isolating susceptibility to the
training shortcut. **OOD also includes transfer effects** because its acquisition groups
were unseen during training.

.. code-block:: python

   from croma import nipd

   score = nipd(
       accuracies=[[0.90, 0.90], [0.70, 0.70], [0.50, 0.50]],
       cramers_v=[0.0, 0.5, 1.0],
       chance=0.5,
   )  # -0.5

Download the exact float basis as `JSON <nipd.json>`_ or the complete tabular view as
`CSV <nipd.csv>`_. Both files are committed publication
artifacts; the static tables below read the JSON directly and require no JavaScript or
local study output.

Published results
-----------------

Each cohort states its balanced-accuracy chance level. The tile tables include the
unranked natural-image control ``DINOv2-B`` (†); PCaBiop contains only whole-slide
encoders. ``CRoMa`` in the downloadable payload is the pooled median at ``m = 5``.

Camelyon
^^^^^^^^

Biological classes: 2; chance balanced accuracy: 0.500.

.. nipd-table:: camelyon id

.. nipd-table:: camelyon ood

TCGA-4×4
^^^^^^^^

Biological classes: 4; chance balanced accuracy: 0.250.

.. nipd-table:: tcga-4x4 id

.. nipd-table:: tcga-4x4 ood

Tolkach-ESCA
^^^^^^^^^^^^

Biological classes: 6; chance balanced accuracy: 0.167.

.. nipd-table:: tolkach-esca id

.. nipd-table:: tolkach-esca ood

PCaBiop
^^^^^^^

Biological classes: 2; chance balanced accuracy: 0.500.

.. nipd-table:: pcabiop id

.. nipd-table:: pcabiop ood

APD continuity
--------------

``APD`` remains available as the PathoROB-faithful continuity reduction. It normalizes
by raw baseline accuracy and averages repeat-specific ratios; nIPD is the primary result
because it normalizes by above-chance skill and integrates over the observed Cramér's-V
coordinates.
