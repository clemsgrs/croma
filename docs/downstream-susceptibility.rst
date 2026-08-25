Downstream shortcut susceptibility
==================================

``nIPD`` is a signed downstream-susceptibility measure. **Negative nIPD means net
degradation** as training-set confounding increases, **near zero means little or no net
change over the confounding range**, and **positive nIPD means net improvement**. A
near-zero signed area can include offsetting changes, so it is not proof of stable
performance at every point.

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
