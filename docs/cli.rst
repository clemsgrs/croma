CLI
===

Installing ``croma`` puts a ``croma`` executable on your path. It scores an embedding
matrix that already exists on disk and prints a JSON payload to stdout -- it does not embed
images.

.. code-block:: bash

   croma --help

Every metric command takes the same four required inputs, plus its own knobs.

Shared arguments
----------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Argument
     - Default
     - Notes
   * - ``--manifest``
     - *required*
     - Path to the manifest CSV.
   * - ``--embeddings``
     - *required*
     - Path to a ``.npy`` of shape ``(N, D)``, row-aligned to the manifest.
   * - ``--confounder-column``
     - *required*
     - Manifest column to treat as the non-biological confounder.
   * - ``--evaluation-design``
     - *required*
     - ``paired_2x2`` or ``dataset_wide``. See :ref:`evaluation-designs`.
   * - ``--k-candidates``
     - ``5,11,21``
     - Comma-separated. The operating ``k`` is chosen from these by biological kNN
       balanced accuracy. Not used by ``croma croma``.

``croma ri``
------------

.. code-block:: bash

   croma ri \
     --manifest manifest.csv \
     --embeddings embeddings.npy \
     --confounder-column center \
     --evaluation-design paired_2x2

``croma mari``
--------------

Adds ``--tau``, the distance-decay temperature.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Argument
     - Default
     - Notes
   * - ``--tau``
     - *auto*
     - Omit it. ``croma`` then uses this dataset's median typed-neighbour distance at the
       operating ``k`` -- the on-scale value. Pass a float only to reproduce a published
       number. See :ref:`choosing-tau`.

.. code-block:: bash

   croma mari \
     --manifest manifest.csv \
     --embeddings embeddings.npy \
     --confounder-column center \
     --evaluation-design paired_2x2

The payload reports which temperature was used and where it came from:

.. code-block:: json

   {
     "k": 11,
     "value": 0.7312,
     "tau": 0.0261,
     "tau_source": "auto",
     "undefined_frac": 0.031
   }

``tau_source`` is ``"auto"`` when ``--tau`` was omitted and ``"fixed"`` when you pinned it.
A pinned value that sits off the dataset's typed-distance scale prints a warning to stderr.

``croma croma``
---------------

Computes CRoMa. It searches outward for typed neighbours rather than using a fixed ``k``,
so it takes no ``--k-candidates``.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Argument
     - Default
     - Notes
   * - ``--m``
     - ``5``
     - Typed neighbours averaged per type. The headline operating point.
   * - ``--alpha``
     - ``0.10``
     - Tail level for ``q_alpha`` and ``ltm_alpha``.
   * - ``--start-k``
     - ``200``
     - Initial neighbour-search radius.
   * - ``--k-growth-factor``
     - ``2.0``
     - Geometric growth factor when the initial radius holds too few typed neighbours.

.. code-block:: bash

   croma croma \
     --manifest manifest.csv \
     --embeddings embeddings.npy \
     --confounder-column center \
     --evaluation-design paired_2x2

Preparing aligned embeddings
----------------------------

Under ``paired_2x2`` the same image often appears in several subsets. Embed each unique
image once, then expand back to manifest-row order. See :ref:`deduplicated-embeddings`.

``croma build-embedding-manifest``
   Writes the deduplicated list of images to embed.

   .. code-block:: bash

      croma build-embedding-manifest \
        --manifest manifest.csv \
        --confounder-column center \
        --out embedding_manifest.csv

``croma expand-embeddings``
   Expands deduplicated embeddings back to one row per manifest row.

   .. code-block:: bash

      croma expand-embeddings \
        --manifest manifest.csv \
        --confounder-column center \
        --embedding-manifest embedding_manifest.csv \
        --embeddings deduped.npy \
        --out embeddings.npy

Passing a deduplicated matrix straight to a metric command is an error: the metrics require
``N == len(manifest)`` and will tell you to run these two commands first.
