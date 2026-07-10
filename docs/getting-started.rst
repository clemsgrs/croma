Getting Started
===============

Install
-------

.. code-block:: bash

   pip install croma

The core package depends only on ``numpy``, ``pandas``, ``scikit-learn``, and ``tqdm``. It
never loads a model or reads an image -- you bring the embeddings.

To also run the paper-reproduction pipeline (embedding extraction, plotting):

.. code-block:: bash

   pip install "croma[repro]"

What you need
-------------

Two things, and they must line up row for row:

1. a **manifest** CSV with one row per sample, carrying its ``label`` and the
   ``confounder`` you want to test against;
2. an **embeddings** array of shape ``(N, D)`` where row ``i`` is the embedding of manifest
   row ``i``.

See :doc:`manifest` for the full contract. In short: don't normalize, don't reorder, and
make ``N == len(manifest)``.

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
       evaluation_design="dataset_wide",
   )

   ri = RI.compute(features, manifest, k_candidates=[5, 11, 21], **common)
   mari = MaRI.compute(features, manifest, k_candidates=[5, 11, 21], **common)
   croma = CRoMa.compute(features, manifest, **common)

   print(f"RI    {ri.value:.3f}  (k={ri.k}, undefined {ri.undefined_frac:.1%})")
   print(f"MaRI  {mari.value:.3f}  (tau={mari.tau:.4f})")
   print(f"CRoMa {croma.value:+.3f}  (lower-tail mean {croma.ltm_alpha:+.3f})")

Reading the numbers
-------------------

- **RI** and **MaRI** live in :math:`[0, 1]`. Above ``0.5`` the biological evidence
  outweighs the confounder evidence.
- **CRoMa** lives in :math:`(-1, 1)` and is neutral at ``0``. Positive is biology-dominant.

Three habits will keep you out of trouble:

1. **Always read** ``undefined_frac`` **next to RI and MaRI.** Samples with no typed
   neighbour in their top ``k`` are excluded from the score. A high RI over a thin support
   is not a strong result.
2. **Never pin** ``tau``. The default resolves it per model, on the scale of that model's
   own neighbour distances. See :ref:`choosing-tau`.
3. **Read the tail, not just the mean.** ``croma.ltm_alpha`` is the mean of the worst 10%
   of samples. Pooled scores hide brittle subgroups.

Next steps
----------

- :doc:`metrics` -- what each metric measures, and when they disagree.
- :doc:`manifest` -- the manifest and embedding contracts, and the two evaluation designs.
- :doc:`cli` -- score a saved embedding matrix without writing Python.
- :doc:`benchmarking` -- compare many models reproducibly.
