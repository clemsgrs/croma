croma
=====

``croma`` measures how much a pathology foundation model's representation is driven by
biology rather than by non-biological technical variation -- staining, scanning, tissue
preparation -- across centers.

It implements three complementary neighbourhood metrics:

- **RI**, the Robustness Index, which counts favourable versus unfavourable neighbours
- **MaRI**, the Margin-aware Robustness Index, which weights that same evidence by distance
- **CRoMa**, the Cross-confounder Robustness Margin, a signed margin that also supports
  tail-aware reporting

RI was introduced in the `PathoROB <https://arxiv.org/abs/2507.17845>`_ study. ``croma``
provides a clean re-implementation of it, adds MaRI as its margin-aware extension, and
introduces CRoMa, which overcomes limitations of both. MaRI and CRoMa are described in
`Beyond counts: A distributional robustness margin for pathology foundation models
<https://arxiv.org/abs/2607.25497>`_.

.. code-block:: bash

   pip install croma

Twenty-one encoders, three cohorts
----------------------------------

.. aggregate-table::
   :top: 8

**Bold** marks the Pareto frontier — the encoders no other encoder beats on both rankings at
once. † is a natural-image control. The two ranks are kept apart on purpose: a strong median
margin can hide a brittle tail, and one combined number would conceal exactly what the tail
column is for — which is why each cohort cell shows both, median ``CRoMa``/``LTM₁₀``.
:doc:`Full panel and per-cohort detail → <results/index>`

.. themed-figure:: _static/figures/rank_pareto
   :alt: Mean CRoMa rank against mean tail rank, one point per encoder, frontier ringed.

   Median margin against tail severity, aggregated by rank across the three cohorts. Better
   is up and to the right; ringed points are undominated.

.. raw:: html

   <div class="croma-card-grid">
     <a class="croma-card" href="getting-started.html">
       <h3>Getting started</h3>
       <p>Install croma and score your first model in a few lines.</p>
     </a>
     <a class="croma-card" href="metrics.html">
       <h3>Metrics</h3>
       <p>What RI, MaRI, and CRoMa measure, and when they disagree.</p>
     </a>
     <a class="croma-card" href="manifest.html">
       <h3>Inputs</h3>
       <p>The manifest and embedding contracts, and the two evaluation designs.</p>
     </a>
     <a class="croma-card" href="results/index.html">
       <h3>Results</h3>
       <p>Every cohort, every column, and the distributions behind them.</p>
     </a>
   </div>

Citing
------

.. code-block:: bibtex

   @article{grisi2026beyond,
     title   = {Beyond counts: A distributional robustness margin for pathology foundation models},
     author  = {Grisi, Cl{\'e}ment and van der Laak, Jeroen and Litjens, Geert},
     journal = {arXiv preprint arXiv:2607.25497},
     year    = {2026},
     doi     = {10.48550/arXiv.2607.25497},
     url     = {https://arxiv.org/abs/2607.25497}
   }

Please cite the `PathoROB <https://arxiv.org/abs/2507.17845>`_ study alongside it: RI is
theirs, and the three cohorts scored under :doc:`Results <results/index>` are theirs too.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting-started
   metrics
   manifest
   cli
   api

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: In Depth

   results/index
   datasets
   benchmarking
