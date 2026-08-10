Getting Started
===============

Install
-------

.. code-block:: bash

   pip install croma

The package depends only on ``numpy``, ``pandas``, ``scikit-learn`` and ``tqdm``. It
never loads a model or reads an image -- you bring the embeddings.

What you need
-------------

Two things, and they must line up row for row:

1. a **manifest** CSV, one row per sample, carrying its ``label`` and the ``confounder``
   you want to test against;
2. an **embeddings** array of shape ``(N, D)`` where row ``i`` is the embedding of manifest
   row ``i``.

Don't normalize, don't reorder, and make ``N == len(manifest)``. Full contract:
:doc:`manifest`.

Score a model
-------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from croma import CRoMa, MaRI, RI

   manifest = pd.read_csv("manifest.csv")
   features = np.load("embeddings.npy")

   common = dict(
       confounder_column="center",
       evaluation_design="all",
   )

   ri = RI.compute(features, manifest, k_candidates=[5, 11, 21], **common)
   mari = MaRI.compute(features, manifest, k_candidates=[5, 11, 21], **common)
   croma = CRoMa.compute(features, manifest, **common)

   print(f"RI    {ri.value:.3f}  (k={ri.k}, undefined {ri.undefined_frac:.1%})")
   print(f"MaRI  {mari.value:.3f}  (tau={mari.tau:.4f})")
   print(f"CRoMa {croma.value:+.3f}  (lower-tail mean {croma.ltm_alpha:+.3f})")

RI and MaRI live in :math:`[0, 1]`, favouring biology above ``0.5``. CRoMa lives in
:math:`(-1, 1)` and is neutral at ``0``.

Before you trust a number
-------------------------

- Read the support beside RI and MaRI: both pool only over the samples with typed
  evidence inside ``k``, so a high score resting on a thin support fraction
  (``1 - undefined_frac``) is not a strong result (:ref:`undefined neighbourhoods
  <undefined-neighbourhoods>`).
- Leave ``tau`` at its default. A fixed ``tau`` shared across models distorts exactly what
  MaRI exists to measure (:ref:`choosing-tau`).
- Read the tail. ``ltm_alpha`` is the mean of the worst 10% of samples; pooled scores hide
  brittle subgroups (:ref:`tail-reporting`).

From here, :doc:`metrics` covers what each metric measures and when they disagree, and
:doc:`manifest` covers the input contracts in full.
