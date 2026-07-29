# croma

<p align="center">
  <a href="https://arxiv.org/abs/2607.25497"><img src="https://img.shields.io/badge/arXiv-2607.25497-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pypi.org/project/croma/"><img src="https://img.shields.io/pypi/v/croma.svg" alt="PyPI version"></a>
  <a href="https://clemsgrs.github.io/croma/"><img src="https://img.shields.io/badge/docs-github.io-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: Black"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/clemsgrs/croma.svg" alt="License"></a>
</p>

`croma` measures how much a pathology foundation model's representation is driven by biology rather than by non-biological technical variation -- staining, scanning, tissue preparation -- across centers.

| Metric | Name | What it does |
| --- | --- | --- |
| `RI` | Robustness Index | Counts favourable vs. unfavourable neighbours |
| `MaRI` | Margin-aware Robustness Index | Weights that same evidence by feature distance |
| `CRoMa` | Cross-confounder Robustness Margin | A signed margin, with tail-aware reporting |

RI was introduced in the [PathoROB](https://arxiv.org/abs/2507.17845) study. `croma` provides a clean re-implementation of it, adds MaRI as its margin-aware extension, and introduces CRoMa, which overcomes limitations of both. MaRI and CRoMa are described in [_Beyond counts: A distributional robustness margin for pathology foundation models_](https://arxiv.org/abs/2607.25497).

For downstream shortcut-susceptibility experiments, `croma` also provides the
confounder-biased probe protocol, PathoROB's APD reduction, and normalized integrated
performance degradation (nIPD). nIPD expresses degradation relative to the
above-chance baseline performance available to lose and integrates it over Cramér's
V. See the [downstream API](https://clemsgrs.github.io/croma/api.html#downstream-reductions).

📖 **[Documentation](https://clemsgrs.github.io/croma/)**

## Install

```bash
pip install croma
```

The core package depends only on `numpy`, `pandas`, `scikit-learn` and `tqdm`. It never loads a model or reads an image -- you bring the embeddings. Add the paper-reproduction utilities with `pip install "croma[repro]"`.

## Quickstart

You need a **manifest** CSV with one row per sample, and an **embeddings** array of shape `(N, D)` whose row `i` is the embedding of manifest row `i`. Don't normalize them -- `croma` L2-normalizes internally and compares neighbours by cosine distance.

```python
import numpy as np
import pandas as pd
from croma import CRoMa, MaRI, RI

manifest = pd.read_csv("manifest.csv")
features = np.load("embeddings.npy")

common = dict(confounder_column="center", evaluation_design="paired_2x2")

ri = RI.compute(features, manifest, k_candidates=[5, 11, 21], **common)
mari = MaRI.compute(features, manifest, k_candidates=[5, 11, 21], **common)
croma = CRoMa.compute(features, manifest, **common)

print(f"RI    {ri.value:.3f}  (k={ri.k}, undefined {ri.undefined_frac:.1%})")
print(f"MaRI  {mari.value:.3f}  (tau={mari.tau:.4f})")
print(f"CRoMa {croma.value:+.3f}  (lower-tail mean {croma.ltm_alpha:+.3f})")
```

## Results

Twenty-one encoders — twenty pathology foundation models and one natural-image control (†) — scored on three tile cohorts from [PathoROB](https://arxiv.org/abs/2507.17845).

<!-- results:start -->
| Model | CRoMa rank | tail rank | Camelyon | TCGA-4×4 | Tolkach-ESCA |
| --- | ---: | ---: | ---: | ---: | ---: |
| **CONCH** | 2.7 | 8.3 | 0.20 | 0.15 | 0.44 |
| **GenBio-PathFM** | 3.0 | 6.3 | 0.19 | 0.16 | 0.39 |
| Midnight-12k | 3.0 | 15.7 | 0.11 | 0.40 | 0.58 |
| **CONCHv1.5** | 3.3 | 4.3 | 0.19 | 0.15 | 0.39 |
| Virchow2 | 4.3 | 5.7 | 0.20 | 0.13 | 0.35 |
| H0-mini | 5.3 | 10.7 | 0.17 | 0.12 | 0.38 |
| Virchow | 6.3 | 9.3 | 0.16 | 0.09 | 0.37 |
| **H-optimus-1** | 8.3 | 3.0 | 0.08 | 0.09 | 0.26 |

Top 8 of 21 by CRoMa rank, over 3 tile cohorts. Each rank is the mean of that encoder's within-cohort ranks — by median CRoMa, and by tail severity LTM₁₀. **Bold** marks the Pareto frontier: the encoders no other encoder beats on both at once. There is deliberately no combined rank, because a strong median can hide a brittle tail.

📊 **[Full panel, per-cohort detail and the distributions](https://clemsgrs.github.io/croma/results/)**
<!-- results:end -->

## Reading the numbers

**RI** and **MaRI** live in `[0, 1]`; above `0.5`, biological evidence outweighs confounder evidence. **CRoMa** lives in `(-1, 1)` and is neutral at `0`, positive when biology dominates.

Three habits will keep you out of trouble:

1. **Always read `undefined_frac` next to RI and MaRI.** Samples with no informative neighbour in their top `k` are excluded from the score, so a high RI over a thin support is not a strong result.
2. **Never pin `tau`.** It defaults to `None`, which resolves it per model on the scale of that model's own neighbour distances. One fixed `tau` shared across models sharpens the margin for some and flattens it for others -- exactly the distortion MaRI exists to remove. See [Choosing tau](https://clemsgrs.github.io/croma/metrics.html#choosing-tau).
3. **Read the tail, not just the mean.** `croma.ltm_alpha` is the mean of the worst 10% of samples. Pooled scores hide brittle subgroups.

## Also in the docs

- **[Evaluation designs](https://clemsgrs.github.io/croma/manifest.html)** -- `paired_2x2` controls what is compared and reports occurrence-level outputs; `dataset_wide` gives one number over the whole cohort at sample level. Includes the manifest contract and minimal valid examples for each.
- **[CLI](https://clemsgrs.github.io/croma/cli.html)** -- the same three metrics from the shell, over a `.npy` that already exists.
- **[Benchmarking](https://clemsgrs.github.io/croma/benchmarking.html)** -- the multi-model pipeline that produced the paper's numbers, split into embed / compute / render steps under `scripts/`.

## Citing

If you use MaRI or CRoMa, please cite the paper — and the
[PathoROB](https://arxiv.org/abs/2507.17845) study that introduced the Robustness Index
these build on.

```bibtex
@article{grisi2026beyond,
  title   = {Beyond counts: A distributional robustness margin for pathology foundation models},
  author  = {Grisi, Cl{\'e}ment and van der Laak, Jeroen and Litjens, Geert},
  journal = {arXiv preprint arXiv:2607.25497},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.25497},
  url     = {https://arxiv.org/abs/2607.25497}
}
```

To cite the software itself rather than the method, use [`CITATION.cff`](CITATION.cff)
directly; GitHub's **Cite this repository** button resolves to the paper above.

## License

[Apache 2.0](LICENSE)
