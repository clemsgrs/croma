# croma

<p align="center">
  <a href="https://pypi.org/project/croma/"><img src="https://img.shields.io/pypi/v/croma.svg" alt="PyPI version"></a>
  <a href="https://clemsgrs.github.io/croma/"><img src="https://img.shields.io/badge/docs-github.io-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: Black"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/clemsgrs/croma.svg" alt="License"></a>
</p>

`croma` measures how much a pathology foundation model's representation is driven by biology rather than by non-biological technical variation -- staining, scanning, tissue preparation -- across centers.

It implements three complementary neighbourhood metrics:

| Metric | Name | What it does |
| --- | --- | --- |
| `RI` | Robustness Index | Counts favourable vs. unfavourable neighbours |
| `MaRI` | Margin-aware Robustness Index | Weights that same evidence by feature distance |
| `CRoMa` | Cross-confounder Robustness Margin | A signed margin, with tail-aware reporting |

RI was introduced in the [PathoROB](https://arxiv.org/abs/2507.17845) study. `croma` provides a clean re-implementation of it, adds MaRI as its margin-aware extension, and introduces CRoMa, which overcomes limitations of both. The package also ships optional benchmarking utilities for multi-model evaluation.

📖 **[Documentation](https://clemsgrs.github.io/croma/)**

## Install

```bash
pip install croma
```

The core package depends only on `numpy`, `pandas`, `scikit-learn` and `tqdm`. It never loads a model or reads an image -- you bring the embeddings.

With paper-reproduction utilities (embedding extraction, plotting):

```bash
pip install "croma[repro]"
```

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

The same three metrics from the shell, reading a `.npy` that already exists:

```bash
croma ri    --manifest manifest.csv --embeddings embeddings.npy \
            --confounder-column center --evaluation-design paired_2x2

croma mari  --manifest manifest.csv --embeddings embeddings.npy \
            --confounder-column center --evaluation-design paired_2x2

croma croma --manifest manifest.csv --embeddings embeddings.npy \
            --confounder-column center --evaluation-design paired_2x2
```

Each prints a JSON payload to stdout. See the [CLI reference](https://clemsgrs.github.io/croma/cli.html).

## Reading the numbers

- **RI** and **MaRI** live in `[0, 1]`. Above `0.5`, biological evidence outweighs confounder evidence.
- **CRoMa** lives in `(-1, 1)` and is neutral at `0`. Positive is biology-dominant.

Three habits will keep you out of trouble:

1. **Always read `undefined_frac` next to RI and MaRI.** Samples with no informative neighbour in their top `k` are excluded from the score, so a high RI over a thin support is not a strong result.
2. **Never pin `tau`.** It defaults to `None`, which resolves it per model on the scale of that model's own neighbour distances. One fixed `tau` shared across models sharpens the margin for some and flattens it for others -- exactly the distortion MaRI exists to remove. See [Choosing tau](https://clemsgrs.github.io/croma/metrics.html#choosing-tau).
3. **Read the tail, not just the mean.** `croma.ltm_alpha` is the mean of the worst 10% of samples. Pooled scores hide brittle subgroups.

## Evaluation designs

Pick `paired_2x2` when you want to control what is being compared: you define subsets in which two labels and two confounder values are all present, so a confounder effect cannot be confused with a class-imbalance effect. This is the PathoROB-style design, and the one the paper reports. It requires a `subset` column and reports **occurrence-level** outputs, since a sample may belong to several subsets.

Pick `dataset_wide` when you want one number over the whole cohort and accept that label and confounder may be unevenly mixed. It needs no `subset` column and reports **sample-level** outputs.

Full contracts, minimal valid manifests for each design, and the deduplicated-embedding workflow are in the [Inputs reference](https://clemsgrs.github.io/croma/manifest.html).

## Benchmarking

`croma` also ships the pipeline used to produce the paper's numbers. It is not part of the installed package; it lives under `scripts/` and is split into three commands along the seams of what is expensive:

```bash
# 1. Embed a tileset once. --manifest is needed only the first time (it derives manifest.csv).
python scripts/bench/extract_embeddings.py \
  --tileset pathorob-camelyon \
  --manifest data/pathorob/manifests/pathorob-camelyon.csv \
  --models UNI,Virchow2

# 2. Compute metrics for a registered benchmark at one operating point.
python scripts/bench/benchmark.py --benchmark camelyon --protocol median-k

# 3. Render that run's figures.
python scripts/bench/render.py output/metrics/median-k/camelyon
```

Every benchmark over a tileset shares its embeddings, so adding an encoder means embedding it once; it then joins every benchmark over that tileset automatically. Benchmarks are declared in `scripts/bench/benchmarks.py`. To sweep them all: `scripts/repro/run_benchmarks.sh median-k`. See [ADR-0007](docs/adr/0007-embeddings-are-a-tileset-benchmarks-are-views.md) and the [benchmarking guide](https://clemsgrs.github.io/croma/benchmarking.html).

## Citing

The paper describing MaRI and CRoMa is in preparation. Until it is out, please cite this
repository, along with the [PathoROB](https://arxiv.org/abs/2507.17845) study that
introduced the Robustness Index.

## License

[Apache 2.0](LICENSE)
