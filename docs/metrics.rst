Metrics
=======

All three metrics ask the same question of every sample: among its nearest neighbours in
feature space, does it sit closer to samples that share its **biology** or to samples that
share its **confounder**? They differ in how they turn that neighbourhood into a number.

.. themed-figure:: _static/figures/concept_metrics
   :alt: One neighbourhood read three ways -- counted by RI, distance-weighted by MaRI,
         and reduced to a signed margin by CRoMa.

   One neighbourhood, read three ways. RI counts typed neighbours inside a fixed ``k``;
   MaRI keeps the window but weights by distance; CRoMa drops the window and compares the
   nearest ``SO`` and ``OS`` distances directly.

Neighbour types
---------------

Fix a sample with label :math:`\ell` and confounder :math:`c`. Each of its neighbours falls
into one of four types:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Type
     - Neighbour
     - Meaning
   * - ``SO``
     - same label, other confounder
     - The **biological match**. A robust model keeps these close.
   * - ``OS``
     - other label, same confounder
     - The **confounder impostor**. A shortcut-prone model keeps these close.
   * - ``SS``
     - same label, same confounder
     - Uninformative: biology and confounder agree.
   * - ``OO``
     - other label, other confounder
     - Uninformative: biology and confounder both differ.

Only ``SO`` and ``OS`` neighbours -- the *typed* neighbours -- carry evidence. Neighbours
from the same slide are always excluded, so a model cannot score well by retrieving other
tiles of the slide it is already looking at.

.. _undefined-neighbourhoods:

.. note::

   A sample whose top-``k`` contains no typed neighbour at all is **undefined** for RI and
   MaRI. Always read ``result.undefined_frac`` alongside the score: a high RI over a
   handful of defined samples is not a strong result. CRoMa avoids this by searching
   outward until it finds typed neighbours.

RI -- Robustness Index
----------------------

RI counts. Within a sample's top ``k``, it weighs ``SO`` neighbours against ``OS``
neighbours and scores the biological share of that typed evidence,
:math:`n_{SO} / (n_{SO} + n_{OS})`. Every neighbour counts the same, whether adjacent to
the sample or at the edge of the neighbourhood.

``result.value`` pools this across the dataset -- total ``SO`` evidence over total typed
evidence -- so it lies in :math:`[0, 1]`: above ``0.5`` biology dominates, below it the
confounder does. Pooling means samples with richer typed neighbourhoods contribute more;
the median of the per-sample scores is reported separately as ``result.median_value``.

``k`` is not a free parameter. It is chosen from ``k_candidates`` as the value maximizing
biological kNN balanced accuracy, so RI and MaRI always operate at the same ``k``.

.. code-block:: python

   from croma import RI

   ri = RI.compute(
       features, manifest,
       confounder_column="center",
       evaluation_design="paired_2x2",
       k_candidates=[5, 11, 21],
   )
   print(ri.value, ri.k, ri.undefined_frac)

MaRI -- Margin-aware Robustness Index
-------------------------------------

Two models can earn the same RI while one keeps its biological matches far closer than its
impostors and the other barely separates them. MaRI recovers that difference by replacing
each neighbour's count of 1 with a weight :math:`\exp(-d / \tau)`, where :math:`d` is its
cosine distance: near neighbours dominate, distant ones fade. Everything else -- the
biological share, the pooling, the :math:`[0, 1]` range, the 0.5 neutral point -- is as in
RI, so RI is exactly MaRI with every weight set to 1.

.. themed-figure:: _static/figures/ri_mari
   :alt: Two models with identical RI separate once neighbours are weighted by distance.

   Same counts, different margins. The two neighbourhoods score identically under RI;
   weighting by distance separates them.

.. code-block:: python

   from croma import MaRI

   mari = MaRI.compute(
       features, manifest,
       confounder_column="center",
       evaluation_design="paired_2x2",
       k_candidates=[5, 11, 21],
   )
   print(mari.value, mari.tau)

.. _choosing-tau:

Choosing ``tau``
~~~~~~~~~~~~~~~~

``tau`` sets how fast that weight decays, so it only means anything relative to the *scale*
of the typed-neighbour distances. Far **below** that scale the score collapses onto the
single nearest typed neighbour -- winner-take-all, and noisy. Far **above** it every weight
is :math:`\approx 1` and MaRI degenerates back into count-based RI, losing the margin
entirely.

That scale is a property of each embedding, not of the dataset: a tight representation's
typed neighbours sit much closer than a diffuse one's. **One fixed** ``tau`` **shared across
models therefore sharpens the margin for some and flattens it for others** -- exactly the
distortion MaRI exists to remove. So ``croma`` resolves it automatically, per model, as the
**median typed-neighbour cosine distance at the operating** ``k``, and reports the value
used on ``result.tau``. Leave it unset.

This is not circular: ``k`` is selected by biological kNN balanced accuracy, which never
consults the weighting, so ``k`` is fixed before ``tau`` is chosen.

You can still pin ``tau`` -- to reproduce a published number, say. ``croma`` warns if the
pinned value sits more than a factor of 4 from the dataset's typed-distance median; silence
that with ``warn_tau=False``. To inspect the recommendation without scoring, call
:meth:`~croma.MaRI.recommend_tau`.

.. warning::

   Pinned ``tau`` values are comparable across models only if those models' typed-neighbour
   distances share a scale. They generally do not.

CRoMa -- Cross-confounder Robustness Margin
-------------------------------------------

RI and MaRI are undefined whenever the top ``k`` happens to contain no typed neighbour, and
both compress a whole neighbourhood into a win/lose verdict. CRoMa instead searches outward
until it finds typed neighbours of both kinds, then reports a signed margin:

.. math::

   \mathrm{CRoMa}_i = \frac{d_{OS} - d_{SO}}{d_{OS} + d_{SO}}

where :math:`d_{SO}` and :math:`d_{OS}` are the mean cosine distances to the ``m`` nearest
``SO`` and ``OS`` neighbours. The result lies in :math:`(-1, 1)`: positive is
**biology-dominant** (the biological match is closer than the impostor), negative is
**confounder-dominant** and fragile, and zero is an exactly contested boundary.
Equivalently, the impostor accounts for a fraction :math:`(1 + \mathrm{CRoMa}_i)/2` of the
total typed distance.

.. themed-figure:: _static/figures/croma_geometry
   :alt: A fragile and a robust model, with their nearest SO and OS distances marked.

   The same anchor under a fragile and a robust model. CRoMa is the normalized difference
   between the impostor distance and the biological-match distance.

.. code-block:: python

   from croma import CRoMa

   croma = CRoMa.compute(
       features, manifest,
       confounder_column="center",
       evaluation_design="paired_2x2",
   )
   print(croma.value, croma.q_alpha, croma.ltm_alpha)

Choosing ``m``
~~~~~~~~~~~~~~

``m`` is the number of typed neighbours averaged per type. The default ``m=5`` is the
headline operating point: ``m=1`` is maximally sensitive to a single outlier neighbour,
while ``m=5`` is the smallest window in which no one neighbour exceeds 20% of the estimate,
without leaving the local typed shell.

Model *rankings* and the biology-/confounder-dominant *sign* are near-invariant across
``m`` -- Spearman :math:`\ge 0.98` across the tile benchmarks -- while ``m`` does move
per-sample magnitudes and the tail statistics. Pass a list to sweep it in one pass:

.. code-block:: python

   results = CRoMa.compute(..., m=[1, 5, 10])   # -> dict[int, CRoMaResult]

.. _tail-reporting:

Tail reporting
--------------

A pooled score hides brittle subgroups: a model can look robust on average while an
identifiable slice of its samples is confounder-dominant. Every CRoMa result therefore
carries two tail statistics at level ``alpha`` (default ``0.10``):

- ``q_alpha`` -- the ``alpha``-quantile of the per-sample CRoMa distribution.
- ``ltm_alpha`` -- the **lower-tail mean**: the mean of the worst ``alpha`` fraction.

Report these next to ``value``. Two models with the same pooled CRoMa can have very
different lower tails, and the lower tail is where deployment risk lives.
