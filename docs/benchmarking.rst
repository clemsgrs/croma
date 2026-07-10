Benchmarking
============

Alongside the library, the repository ships the pipeline used to produce the paper's
numbers. It is **not** part of the installed package -- it lives under ``scripts/`` in the
`source tree <https://github.com/clemsgrs/croma>`_ and needs the ``repro`` extra:

.. code-block:: bash

   pip install "croma[repro]"

Three commands, split along the seams of what is expensive
----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Does
   * - ``extract_embeddings.py``
     - Embeds a **tileset** once, into ``output/embeddings/<tileset>/``.
   * - ``benchmark.py``
     - Computes metrics for one **benchmark** (a row-view of a tileset) at one
       **protocol**. Reads embeddings, never writes any. Results land in
       ``output/metrics/<protocol>/<benchmark>/``.
   * - ``render.py``
     - Renders a run's figure set from the written artifacts.

.. code-block:: bash

   # 1. Embed a tileset once. --manifest is needed only the first time (it derives manifest.csv).
   python scripts/bench/extract_embeddings.py \
     --tileset pathorob-camelyon \
     --manifest data/pathorob/manifests/pathorob-camelyon.csv \
     --models UNI,Virchow2

   # 2. Compute metrics for a registered benchmark at one operating point.
   python scripts/bench/benchmark.py --benchmark camelyon --protocol median-k

   # 3. Render that run's figures.
   python scripts/bench/render.py output/metrics/median-k/camelyon

Embeddings are a tileset; benchmarks are views
----------------------------------------------

Every benchmark over a tileset shares that tileset's embeddings. Adding an encoder means
embedding it once; it then joins every benchmark over that tileset automatically. Benchmarks
are declared in ``scripts/bench/benchmarks.py``.

To sweep them all:

.. code-block:: bash

   scripts/repro/run_benchmarks.sh median-k

The rationale is recorded in
`ADR-0007 <https://github.com/clemsgrs/croma/blob/main/docs/adr/0007-embeddings-are-a-tileset-benchmarks-are-views.md>`_.

``tau`` across models
---------------------

``benchmark.py`` resolves ``tau`` per model by default -- each model gets the median typed
neighbour distance of *its own* embedding at the operating ``k``. This matches the library
default and is the only setting under which MaRI is comparable across models. Passing
``--tau <float>`` pins one temperature for every model; the run then prints which models it
is off-scale for. See :ref:`choosing-tau`.
