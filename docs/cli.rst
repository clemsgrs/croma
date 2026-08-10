CLI
===

Installing ``croma`` puts a ``croma`` executable on your path. It scores an embedding
matrix that already exists on disk and prints a JSON payload to stdout -- it does not embed
images.

Every metric command takes the same shape:

.. code-block:: bash

   croma {ri|mari|croma} \
     --manifest manifest.csv \
     --embeddings embeddings.npy \
     --confounder-column center \
     --evaluation-design paired_2x2

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
     - ``all``
     - ``all`` or ``paired_2x2``. Omit it to score every manifest row together;
       ``paired_2x2`` has to be asked for. See :ref:`evaluation-designs`.
   * - ``--k-candidates``
     - ``5,11,21``
     - Comma-separated. The operating ``k`` is chosen from these by biological kNN
       balanced accuracy. Not used by ``croma croma``.

``croma ri``
------------

No further arguments.

``croma mari``
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Argument
     - Default
     - Notes
   * - ``--tau``
     - *auto*
     - Omit it. ``croma`` then uses the data's median typed-neighbour distance at the
       operating ``k`` -- the on-scale value. Pass a float only to reproduce a published
       number. See :ref:`choosing-tau`.

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
A pinned value off the dataset's typed-distance scale prints a warning to stderr.

``croma croma``
---------------

Searches outward for typed neighbours rather than using a fixed ``k``, so it takes no
``--k-candidates``.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Argument
     - Default
     - Notes
   * - ``--m``
     - ``5``
     - Typed neighbours averaged per type. The headline radius.
   * - ``--alpha``
     - ``0.10``
     - Tail level for ``q_alpha`` and ``ltm_alpha``.
   * - ``--start-k``
     - ``200``
     - Initial neighbour-search radius.
   * - ``--k-growth-factor``
     - ``2.0``
     - Geometric growth factor when the initial radius holds too few typed neighbours.

The payload adds the distributional statistics to the pooled value:

.. code-block:: json

   {
     "m": 5,
     "value": 0.1842,
     "q_alpha": -0.2013,
     "ltm_alpha": -0.3106,
     "f0": 0.1823,
     "undefined_frac": 0.004
   }

``f0`` is the confounder-dominant fraction :math:`F(0)`; its boundary and denominator are
defined under :ref:`confounder-dominant-fraction`.

Preparing aligned embeddings
----------------------------

Under ``paired_2x2`` the same image often appears in several subsets. Embed each unique
image once, then expand back to manifest-row order (see :ref:`deduplicated-embeddings`):

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

Passing a deduplicated matrix straight to a metric command is an error: the metrics require
``N == len(manifest)`` and will tell you to run these two commands first.
