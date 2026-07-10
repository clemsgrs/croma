API Reference
=============

Reference for the Python API. See :doc:`getting-started` for introductory examples and
:doc:`metrics` for what each metric measures.

``croma`` exposes three metric classes and one alignment helper. Each metric is a namespace
of classmethods -- you never instantiate them. The short names are the ones to import:

.. code-block:: python

   from croma import CRoMa, MaRI, RI

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
