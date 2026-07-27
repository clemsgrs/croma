API Reference
=============

Reference for the Python API. See :doc:`getting-started` for introductory examples and
:doc:`metrics` for what each metric measures.

``croma`` exposes three metric classes, two downstream reductions and one alignment
helper. Each metric class is a namespace of classmethods -- you never instantiate them.
The short names are the ones to import:

.. code-block:: python

   from croma import CRoMa, MaRI, RI, apd, napd

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
   * - ``napd``
     - :func:`~croma.napd` (a function, not a metric namespace)
     - ``float``

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

Downstream reductions
---------------------

Unlike the three metrics above, ``apd`` and ``napd`` read no embeddings: they reduce the
balanced accuracies a confounder-biased probe sweep already produced. Both take the same
``(n_splits, n_iterations)`` matrix, with the balanced baseline in row ``0`` and each
later row a progressively more confounded split.

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

nAPD
^^^^

``napd`` divides by *skill* -- balanced accuracy above chance -- instead of raw accuracy,
which makes drops comparable across tasks with different class counts. So it additionally
needs the task's chance level, ``1 / n_biological_classes``.

.. code-block:: python

   from croma import napd

   napd(accuracies, chance=1 / 2)  # -> e.g. -0.25

.. autofunction:: croma.napd

The two also differ in reduction order: ``apd`` takes the ratio per replicate and averages
afterwards, while ``napd`` averages the replicates first. That difference is deliberate and
is not reconciled -- ``apd``'s order is PathoROB's, and changing it would cost the very
faithfulness the function exists to provide.

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
