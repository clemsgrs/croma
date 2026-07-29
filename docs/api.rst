API Reference
=============

Reference for the Python API. See :doc:`getting-started` for introductory examples and
:doc:`metrics` for what each metric measures.

``croma`` exposes three metric classes, the downstream probe protocol and two
reductions, and one alignment helper. Each metric class is a namespace of classmethods --
you never instantiate them. The short names are the ones to import:

.. code-block:: python

   from croma import CRoMa, MaRI, RI, apd, nipd, probe_sweep

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - Import as
     - Class
     - Returns
   * - ``RI``
     - :class:`~croma.RobustnessIndex`
     - :class:`~croma.types.RobustnessResult`
   * - ``MaRI``
     - :class:`~croma.MarginAwareRobustnessIndex`
     - :class:`~croma.types.RobustnessResult`
   * - ``CRoMa``
     - :class:`~croma.CrossConfounderRobustnessMargin`
     - :class:`~croma.types.CRoMaResult`
   * - ``apd``
     - :func:`~croma.apd` (a function, not a metric namespace)
     - ``float``
   * - ``nipd``
     - :func:`~croma.nipd` (a function, not a metric namespace)
     - ``float``
   * - ``probe_sweep``
     - :func:`~croma.probe_sweep` (a function, not a metric namespace)
     - ``numpy.ndarray``

RI
--

.. autoclass:: croma.RobustnessIndex
   :members: compute, compute_curve

MaRI
----

.. autoclass:: croma.MarginAwareRobustnessIndex
   :members: compute, compute_curve, recommend_tau

CRoMa
-----

.. autoclass:: croma.CrossConfounderRobustnessMargin
   :members: compute

Probe protocol
--------------

``probe_sweep`` produces the matrix the reductions below consume. It trains a probe to
predict the biological class from frozen embeddings while a schedule walks the training
set from balanced to fully confounded, and scores each probe on test rows that do not
move. It takes embeddings and a split assignment: no model is loaded, no manifest read and
no output layout touched.

.. code-block:: python

   from croma import apd, probe_sweep
   from croma.downstream import pathorob_schedule

   accuracies = probe_sweep(
       embeddings,                  # (n_rows, n_features)
       centre_index,                # (n_rows,) confounder index per row
       class_index,                 # (n_rows,) biological class index per row
       schedule=pathorob_schedule("camelyon", rows_per_slide=300),
       rows_per_slide=300,
   )
   apd(accuracies)

.. autofunction:: croma.probe_sweep

Scoring unseen confounders
^^^^^^^^^^^^^^^^^^^^^^^^^^

PathoROB scores every probe both on held-out rows of the confounders it trained on and on
an unseen confounder. Both come off one training pass, so they are one call. This form also
takes ``arrange_slides``, which replaces the one step of a replicate that decides which
slides a split trains on and which sit in the held-out tail -- for a cohort whose slides
cannot be ordered freely. Its default is the sweep's own shuffle, which is the reference
protocol.

.. autofunction:: croma.downstream.probe_sweep_over_test_sets

PathoROB's schedules
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: croma.downstream.pathorob_schedule

These two, and ``croma.downstream.IN_DOMAIN`` -- the key the sweep's own held-out matrix
comes back under -- are reachable under ``croma.downstream`` but are not promoted to the
top level, so they carry no stability promise: minimal-first, per ADR-0002.

Downstream reductions
---------------------

Unlike the three metrics above, ``apd`` and ``nipd`` read no embeddings: they reduce the
balanced accuracies a confounder-biased probe sweep already produced. Both take the same
``(n_splits, n_iterations)`` matrix, with the balanced baseline in row ``0`` and each
later row a progressively more confounded split. ``nipd`` additionally takes the
Cramér's-``V`` coordinate of every row.

APD
^^^

``apd`` is **PathoROB's** metric, reported as the faithful reference. Its reduction is
vendored verbatim from PathoROB (BSD 3-Clause, © 2025 BIFOLD Pathomics; see the
distribution's ``NOTICE``) rather than reimplemented, so a value reported as "APD" is the
value PathoROB would report. It takes no ``chance`` argument: it normalizes by raw
accuracy.

.. code-block:: python

   from croma import apd

   apd(accuracies)  # -> e.g. -0.046

.. autofunction:: croma.apd

nIPD
^^^^

``nipd`` is the normalized integrated performance degradation: the signed area under
the chance-normalized degradation curve over Cramér's ``V``. It divides performance
changes by baseline *skill* -- balanced accuracy above chance -- rather than by raw
baseline accuracy. This corrects unequal baseline headroom across models and tasks.

For mean balanced accuracy across repeated training runs, :math:`\bar a(V)`, baseline
:math:`a_0 = \bar a(0)` and chance :math:`\pi`,

.. math::

   g(V) = \frac{\bar a(V) - a_0}{a_0 - \pi},
   \qquad
   \operatorname{nIPD} = \int_0^1 g(V)\,dV.

The integral is estimated by the trapezoidal rule at the supplied Cramér's-``V``
coordinates. Consequently, interval widths -- not the number of sampled conditions --
weight the curve. The coordinates must be finite, strictly increasing, aligned with the
accuracy rows and span ``0`` to ``1``. The mean baseline must exceed chance; there is no
additional weak-skill threshold.

.. code-block:: python

   from croma import nipd

   nipd(
       accuracies=[
           [0.90, 0.90],
           [0.70, 0.70],
           [0.50, 0.50],
       ],
       cramers_v=[0.0, 0.5, 1.0],
       chance=0.5,
   )  # -> -0.5

.. autofunction:: croma.nipd

For reproducibility, :math:`\bar a(V)` is formed before normalization: nIPD therefore
uses a ratio of repeat means. ``apd`` retains PathoROB's mean-of-repeat-specific-ratios
order because changing it would break faithfulness to the reference implementation.

Alignment
---------

.. autofunction:: croma.expand_features_to_manifest

Result types
------------

.. autoclass:: croma.types.RobustnessResult
   :members:
   :undoc-members:

.. autoclass:: croma.types.CRoMaResult
   :members:
   :undoc-members:
