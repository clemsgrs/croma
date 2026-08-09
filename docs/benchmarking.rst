Benchmarking
============

Alongside the library, the repository ships the pipeline used to produce the paper's
numbers. It is **not** part of the installed package -- it lives under ``scripts/`` in the
`source tree <https://github.com/clemsgrs/croma>`_ and needs the ``repro`` extra:

.. code-block:: bash

   pip install "croma[repro]"

To *embed* a tileset you also need the encoder sources, which are only distributed as git
repositories and so cannot be pinned in the package metadata:

.. code-block:: bash

   pip install -r scripts/bench/requirements-encoders.txt

Scoring embeddings that already exist needs none of them.

Mascaret and Phaet
------------------

The pinned Waiv checkpoints use a Transformers 5 remote-code runtime. Reproduce their
public embedding contracts in a dedicated environment so that this constraint does not
change the runtime used by the other encoders:

.. code-block:: bash

   python -m venv .venv-waiv
   .venv-waiv/bin/pip install -e ".[dev]" \
     -r scripts/bench/requirements-waiv.txt
   .venv-waiv/bin/python scripts/bench/extract_embeddings.py \
     --tileset pathorob-camelyon \
     --manifest data/pathorob/manifests/pathorob-camelyon.csv \
     --models Mascaret,Phaet

``requirements-waiv.txt`` constrains Transformers to ``>=5.14,<6`` and directly lists
only the image/model runtime. It deliberately has no dependency on slide2vec. The models
are loaded from ``wearewaiv/mascaret`` and ``wearewaiv/phaet`` at the immutable revisions
recorded in the model registry; both use FP32 inference and their checkpoint configuration's
``pixel_mean`` and ``pixel_std``.

An opt-in real-weight smoke test verifies repeatable finite outputs and unit L2 norms for
both checkpoints. It can download gated weights and is therefore excluded from the default,
offline test run:

.. code-block:: bash

   CROMA_RUN_WAIV_SMOKE=1 .venv-waiv/bin/python -m pytest \
     tests/test_waiv_smoke.py -q

RudolfV 2
----------

The pinned RudolfV 2 family also uses a Transformers 5 remote-code runtime. Its published
implementation requires timm, so reproduce it in a dedicated environment without
slide2vec:

.. code-block:: bash

   python -m venv .venv-rudolfv2
   .venv-rudolfv2/bin/pip install -e ".[dev]" \
     -r scripts/bench/requirements-rudolfv2.txt
   .venv-rudolfv2/bin/python scripts/bench/extract_embeddings.py \
     --tileset pathorob-camelyon \
     --manifest data/pathorob/manifests/pathorob-camelyon.csv \
     --models "RudolfV 2,RudolfV 2-B,RudolfV 2-S"

``requirements-rudolfv2.txt`` constrains Transformers to ``>=5.14,<6`` and includes the
remote code's timm dependency. The three immutable revisions live in the model registry;
all use FP32 inference and the released 224 px preprocessing contract.

An opt-in gated-weight smoke test verifies exact published pooling, deterministic repeated
inference, and the three output dimensions. The default offline suite skips it:

.. code-block:: bash

   CROMA_RUN_RUDOLFV2_SMOKE=1 .venv-rudolfv2/bin/python -m pytest \
     tests/test_rudolfv2_smoke.py -q

Three commands, split along the seams of what is expensive
----------------------------------------------------------

.. code-block:: bash

   # 1. Embed a tileset once, into output/embeddings/<tileset>/.
   #    --manifest is needed only the first time; it derives manifest.csv.
   python scripts/bench/extract_embeddings.py \
     --tileset pathorob-camelyon \
     --manifest data/pathorob/manifests/pathorob-camelyon.csv \
     --models UNI,Virchow2

   # 2. Compute metrics for one benchmark at one protocol. Reads embeddings,
   #    never writes any. Results land in output/metrics/<protocol>/<benchmark>/.
   python scripts/bench/benchmark.py --benchmark camelyon --protocol median-k

   # 3. Render that run's figure set from the written artifacts.
   python scripts/bench/render.py output/metrics/median-k/camelyon

Each ``<Model>.npy`` and ``<Model>.npy.json`` sidecar is one artifact. The sidecar is
published last as the completion marker and records the checkpoint revision (an immutable
SHA for pinned models, otherwise an explicit null), extraction-contract version, precision,
batch size, output dtype and shape, and tileset-manifest fingerprint. Re-running extraction
skips a pair only when that complete contract still matches. An orphaned, malformed, stale,
or mismatched pair fails loudly; inspect the mismatch and pass ``--force`` only when you
intend to replace it.

Embeddings are a tileset; benchmarks are views
----------------------------------------------

A **benchmark** is a row-view of a **tileset**, and every benchmark over a tileset shares
that tileset's embeddings. Adding an encoder means embedding it once; it then joins every
benchmark over that tileset automatically. Benchmarks are declared in
``scripts/bench/benchmarks.py``, and ``scripts/bench/run_benchmarks.sh median-k`` sweeps
them all. The rationale is recorded in
`ADR-0007 <https://github.com/clemsgrs/croma/blob/main/docs/adr/0007-embeddings-are-a-tileset-benchmarks-are-views.md>`_.

``tau`` across models
---------------------

``benchmark.py`` resolves ``tau`` per model by default -- each model gets the median typed
neighbour distance of *its own* embedding at the operating ``k``. This matches the library
default and is the only setting under which MaRI is comparable across models. Passing
``--tau <float>`` pins one temperature for every model; the run then prints which models it
is off-scale for. See :ref:`choosing-tau`.
