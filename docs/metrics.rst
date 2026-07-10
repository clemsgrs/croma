Metrics
=======

All three metrics ask the same question of every sample: among its nearest neighbours in
feature space, does it sit closer to samples that share its **biology** or to samples that
share its **confounder**? They differ in how they turn that neighbourhood into a number.

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

.. note::

   A sample whose top-``k`` contains no typed neighbour at all is **undefined** for RI and
   MaRI. Always read ``result.undefined_frac`` alongside the score: a high RI over a
   handful of defined samples is not a strong result. CRoMa avoids this by searching
   outward until it finds typed neighbours.

RI -- Robustness Index
----------------------

RI counts. Within a sample's top ``k``, it counts ``SO`` neighbours against ``OS``
neighbours; the sample's score is the biological share of that typed evidence,
:math:`n_{SO} / (n_{SO} + n_{OS})`. Every neighbour in the top ``k`` counts the same,
whether it sits right next to the sample or at the edge of the neighbourhood.

``result.value`` pools this across the dataset -- total ``SO`` evidence over total typed
evidence -- so it lies in :math:`[0, 1]`:

- :math:`> 0.5` -- biology-dominant.
- :math:`= 0.5` -- biological and confounder evidence exactly balanced.
- :math:`< 0.5` -- confounder-dominant.

Because it pools, samples with richer typed neighbourhoods contribute more. The median of
the per-sample scores is reported separately as ``result.median_value``.

``k`` is not a free parameter: it is chosen from ``k_candidates`` as the value maximizing
biological kNN balanced accuracy. RI and MaRI therefore always operate at the same ``k``.

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
cosine distance. Near neighbours dominate; distant ones fade. Everything else -- the
biological share, the pooling, the :math:`[0, 1]` range, the 0.5 neutral point -- is as in
RI, so RI is exactly MaRI with every weight set to 1.

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
of the typed-neighbour distances:

- ``tau`` far **below** that scale collapses the score onto the single nearest typed
  neighbour -- winner-take-all, and noisy.
- ``tau`` far **above** it makes every weight :math:`\approx 1`, and MaRI degenerates back
  into the count-based RI. The margin information is lost.

That scale is a property of each embedding, not of the dataset. A tight representation's
typed neighbours sit much closer than a diffuse one's, so **one fixed** ``tau`` **shared
across models sharpens the margin for some and flattens it for others**, which is exactly
the distortion MaRI exists to remove.

So ``croma`` resolves ``tau`` automatically, per model. Leave it unset:

.. code-block:: python

   mari = MaRI.compute(..., tau=None)   # the default

Auto-``tau`` takes the **median typed-neighbour cosine distance at the operating** ``k``.
This is not circular: ``k`` is selected by biological kNN balanced accuracy, which never
consults the weighting, so ``k`` is fixed before ``tau`` is chosen. The value actually used
is reported on ``result.tau``.

You can still pin ``tau`` -- to reproduce a published number, say. ``croma`` will warn if
the value you pin sits more than a factor of 4 away from the dataset's typed-distance
median; silence that with ``warn_tau=False``. To inspect the recommendation without
scoring, call :meth:`~croma.MaRI.recommend_tau`.

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
``SO`` and ``OS`` neighbours. The result lies in :math:`(-1, 1)`:

- :math:`> 0` -- **biology-dominant**: the biological match is closer than the impostor.
- :math:`< 0` -- **confounder-dominant**: the impostor is closer. Fragile.
- :math:`= 0` -- an exactly contested boundary.

Equivalently, the impostor accounts for a fraction :math:`(1 + \mathrm{CRoMa}_i)/2` of the
total typed distance.

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

``m`` is the number of typed neighbours averaged per type. The default, ``m=5``, is the
headline operating point: ``m=1`` is the single-neighbour estimate and is maximally
sensitive to one outlier neighbour, while ``m=5`` is the smallest window in which no single
neighbour exceeds 20% of the estimate, without leaving the local typed shell.

Model *rankings* and the biology-/confounder-dominant *sign* are essentially invariant
across ``m`` (Spearman :math:`\ge 0.99`); ``m`` moves per-sample magnitudes and the tail
statistics. Pass a list to sweep it in one pass:

.. code-block:: python

   results = CRoMa.compute(..., m=[1, 5, 10])   # -> dict[int, CRoMaResult]

Tail reporting
--------------

A pooled score hides brittle subgroups: a model can look robust on average while an
identifiable slice of its samples is confounder-dominant. Every CRoMa result therefore
carries two tail statistics at level ``alpha`` (default ``0.10``):

- ``q_alpha`` -- the ``alpha``-quantile of the per-sample CRoMa distribution.
- ``ltm_alpha`` -- the **lower-tail mean**: the mean of the worst ``alpha`` fraction.

Report these next to ``value``. Two models with the same pooled CRoMa can have very
different lower tails, and the lower tail is where deployment risk lives.
