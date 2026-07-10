Inputs
======

Every metric takes the same two inputs: a **manifest** describing the samples, and an
**embedding matrix** holding one feature vector per manifest row.

.. _embedding-contract:

Embedding contract
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Property
     - Requirement
   * - Type
     - ``numpy.ndarray``. The CLI loads it with :func:`numpy.load`, so the file must be a
       ``.npy`` holding a plain array -- not an ``.npz`` archive and not a pickled object.
   * - Shape
     - Exactly 2-D, ``(N, D)``. ``N`` is the number of samples, ``D`` the embedding
       dimension. A 1-D or 3-D array is rejected.
   * - ``N``
     - Must equal ``len(manifest)``. Row ``i`` of the array is the embedding of row ``i`` of
       the manifest. Nothing else aligns them -- not ``sample_id``, not ``image_path``.
   * - ``D``
     - Any positive width. Models with different ``D`` are never compared within one call.
   * - dtype
     - Any floating dtype; ``float32`` and ``float64`` both work. ``float32`` is the
       sensible default and is what the benchmark pipeline writes.
   * - Normalization
     - **Do not normalize.** ``croma`` L2-normalizes internally and compares neighbours by
       cosine distance. Pre-normalizing is harmless but redundant.
   * - Missing values
     - Not supported, and not silently tolerated: a ``NaN`` or ``inf`` anywhere in the
       matrix raises ``ValueError`` from the neighbour search.

Row order is the whole contract. If you produced embeddings by deduplicating repeated
images, expand them back to manifest-row order first -- see :ref:`deduplicated-embeddings`.

.. code-block:: python

   import numpy as np
   import pandas as pd

   manifest = pd.read_csv("manifest.csv")
   features = np.load("embeddings.npy")

   assert features.ndim == 2
   assert features.shape[0] == len(manifest)

Manifest contract
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Column
     - Required
     - Notes
   * - ``sample_id``
     - yes
     - Unique identifier for the sample.
   * - ``image_path``
     - yes
     - Path to the source image. Used for provenance and deduplication, never read by the
       metrics themselves.
   * - ``label``
     - yes
     - The biological class. This is the signal a robust model should organize by.
   * - ``slide_id``
     - yes
     - Slide of origin. Neighbours from the *same slide* are excluded from every
       neighbourhood, so a model cannot score well by retrieving other tiles of the slide
       it is already looking at. Give each sample a distinct ``slide_id`` only if the
       samples genuinely come from distinct slides.
   * - *confounder*
     - yes
     - The non-biological factor to test against -- center, scanner, stain protocol. You
       choose the column name and pass it as ``confounder_column=`` / ``--confounder-column``.
   * - ``subset``
     - only for ``paired_2x2``
     - Which 2x2 evaluation subset the row belongs to. See below.
   * - ``dataset``
     - no
     - Free-text name echoed back on the result. Defaults to ``"dataset"``.

.. _evaluation-designs:

Evaluation designs
------------------

**Which do I pick?**

Use ``paired_2x2`` when you want to control what is being compared: you hand-build subsets
where two labels and two confounder values are all present, so a confounder effect cannot
be confused with a class-imbalance effect. This is the PathoROB-style design and the one
the paper reports.

Use ``dataset_wide`` when you want a single number over the whole cohort and you accept
that label and confounder may be unevenly mixed. It needs no ``subset`` column, so it is
the right first thing to run on a new dataset.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - ``paired_2x2``
     - ``dataset_wide``
   * - ``subset`` column
     - required
     - ignored
   * - Neighbourhood scope
     - within each subset
     - the whole retained dataset
   * - Evaluated once per
     - ``(sample, subset)`` occurrence
     - sample
   * - ``result.evaluation_unit``
     - ``"occurrence"``
     - ``"sample"``

That last row is the one that surprises people. Under ``paired_2x2`` a sample may belong to
several subsets, so it contributes one *occurrence* to each; the reported score is an
average over occurrences, not over samples. Under ``dataset_wide`` each sample is scored
exactly once. So ``result.n_pairs`` counts occurrences in the first case and samples in the
second, and the two designs are not directly comparable.

``paired_2x2``
~~~~~~~~~~~~~~

Each ``subset`` value names one group of rows. A subset is used only if it forms a
**complete 2x2**: exactly two distinct ``label`` values, exactly two distinct confounder
values, and at least one sample in each of the four ``(label, confounder)`` cells.

.. warning::

   Subsets that are not complete 2x2 are **silently skipped**, not reported as errors. If
   your scores come from fewer occurrences than you expected, check subset completeness
   first. If *no* subset is complete, the call raises.

A minimal valid manifest -- one subset, four cells, one sample each:

.. code-block:: text

   sample_id,image_path,label,slide_id,center,subset
   s1,/data/1.png,tumor,slide-1,center_a,tumor_vs_normal__a_b
   s2,/data/2.png,tumor,slide-2,center_b,tumor_vs_normal__a_b
   s3,/data/3.png,normal,slide-3,center_a,tumor_vs_normal__a_b
   s4,/data/4.png,normal,slide-4,center_b,tumor_vs_normal__a_b

A row carries exactly one ``subset`` value. To evaluate the same sample inside several 2x2
comparisons, repeat the row once per subset -- and repeat its embedding row to match.
``scripts/prep/prepare_paired_manifest.py`` expands a flat manifest into all complete 2x2
combinations for you.

``dataset_wide``
~~~~~~~~~~~~~~~~

No ``subset`` column. Every row is scored against the whole cohort:

.. code-block:: text

   sample_id,image_path,label,slide_id,center
   s1,/data/1.png,tumor,slide-1,center_a
   s2,/data/2.png,tumor,slide-2,center_b
   s3,/data/3.png,normal,slide-3,center_a
   s4,/data/4.png,normal,slide-4,center_b

.. _deduplicated-embeddings:

Deduplicated embeddings
-----------------------

Because ``paired_2x2`` repeats rows across subsets, the same image is often embedded more
than once. Embed each unique image once, then expand back to manifest-row order:

.. code-block:: bash

   croma build-embedding-manifest \
     --manifest manifest.csv \
     --confounder-column center \
     --out embedding_manifest.csv

   # ... embed the rows of embedding_manifest.csv into deduped.npy ...

   croma expand-embeddings \
     --manifest manifest.csv \
     --confounder-column center \
     --embedding-manifest embedding_manifest.csv \
     --embeddings deduped.npy \
     --out embeddings.npy

The equivalent Python entry point is :func:`croma.expand_features_to_manifest`.
