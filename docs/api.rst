API Reference
=============

Reference for the Python API. See :doc:`getting-started` for introductory examples and
:doc:`metrics` for what each metric measures.

``croma`` exposes three metric classes, one downstream reduction and one alignment helper.
Each metric class is a namespace of classmethods -- you never instantiate them. The short
names are the ones to import:

.. code-block:: python

   from croma import CRoMa, MaRI, RI, napd

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

nAPD
----

Unlike the three metrics above, ``napd`` reads no embeddings: it reduces the balanced
accuracies a confounder-biased probe sweep already produced. Pass the
``(n_splits, n_iterations)`` matrix with the balanced baseline in row ``0``, and the
task's chance level ``1 / n_biological_classes``.

.. code-block:: python

   from croma import napd

   napd(accuracies, chance=1 / 2)  # -> e.g. -0.25

.. autofunction:: croma.napd

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
