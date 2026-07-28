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
introduces CRoMa, which overcomes limitations of both.

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
     <a class="croma-card" href="cli.html">
       <h3>CLI</h3>
       <p>Score a saved embedding matrix straight from the shell.</p>
     </a>
   </div>

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

   datasets
   benchmarking
