# croma

<p align="center">
  <a href="https://pypi.org/project/cross-margin/"><img src="https://img.shields.io/pypi/v/cross-margin.svg" alt="PyPI version"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: Black"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/clemsgrs/croma.svg" alt="License"></a>
</p>

`croma` is a Python package for evaluating pathology foundation model robustness to non-biological confounders.

It implements three complementary metrics:

- `RI`: Robustness Index
- `MaRI`: Margin-aware Robustness Index
- `CCMR`: Cross-Confounder Margin Ratio

`croma` provides a clean implementation of RI together with MaRI -- its margin-aware extension -- as well as a new robustness metric, CCMR -- which overcomes some of the limitations of RI/MaRI and enables tail-aware analysis for more detailed robustness characterization. RI was originally introduced in the [PathoROB](https://arxiv.org/abs/2507.17845) study.


The package also comes with optional benchmarking utilities for multi-model evaluation.

## Install

Core package:

```bash
pip install cross-margin
```

With benchmark utilities:

```bash
pip install "cross-margin[bench]"
```

## Python Quickstart

```python
import numpy as np
import pandas as pd
from croma import CCMR, MaRI, RI

manifest = pd.read_csv("manifest.csv")
features = np.load("embeddings.npy")

ri = RI.compute(
    features,
    manifest,
    confounder_column="confounder",
    evaluation_design="paired_2x2",
    k_candidates=[5, 11, 21],
)

mari = MaRI.compute(
    features,
    manifest,
    confounder_column="confounder",
    evaluation_design="paired_2x2",
    k_candidates=[5, 11, 21],
    tau=0.2,
)

ccmr = CCMR.compute(
    features,
    manifest,
    confounder_column="confounder",
    evaluation_design="paired_2x2",
    m=1,
)
```

## CLI Quickstart

Compute RI:

```bash
croma ri \
  --manifest /path/to/manifest.csv \
  --embeddings /path/to/embeddings.npy \
  --confounder-column confounder \
  --evaluation-design paired_2x2 \
  --k-candidates 5,11,21
```

Compute MaRI:

```bash
croma mari \
  --manifest /path/to/manifest.csv \
  --embeddings /path/to/embeddings.npy \
  --confounder-column confounder \
  --evaluation-design paired_2x2 \
  --k-candidates 5,11,21 \
  --tau 0.2
```

Compute CCMR:

```bash
croma ccmr \
  --manifest /path/to/manifest.csv \
  --embeddings /path/to/embeddings.npy \
  --confounder-column confounder \
  --evaluation-design paired_2x2 \
  --m 1
```

## Benchmarking

`cross-margin` also includes an optional benchmarking pipeline for comparing multiple foundation models on the same manifest. The benchmark handles embedding extraction, metric computation, cached re-runs, and summary artifact generation in a single workflow.

Run it with:

```bash
python scripts/benchmark.py \
  --manifest /path/to/manifest.csv \
  --confounder-column confounder \
  --output-dir /path/to/benchmark
```

For benchmark options, outputs, caching behavior, and downstream analysis, see [docs/benchmarking.md](docs/benchmarking.md).

## Manifest Contract

Required columns:

- `sample_id`
- `image_path`
- `label`
- `slide_id`
- the user-selected confounder column passed as `confounder_column=` or `--confounder-column`

Default contract:

- row `i` in `embeddings.npy` should match row `i` in the manifest

Optional:

- `subset`: required for `evaluation_design="paired_2x2"`

## Evaluation Designs

- `paired_2x2`: uses explicit manifest-defined subsets and reports occurrence-level outputs
- `dataset_wide`: evaluates the retained dataset once and reports sample-level outputs

`paired_2x2` is strict by design: the manifest must define valid 2x2 `(label x confounder)` subsets via the `subset` column.

## Documentation

- [Metric usage](docs/metrics.md)
- [Paired evaluation and PathoROB-style inputs](docs/paired_evaluation.md)
- [Benchmarking and analysis](docs/benchmarking.md)
- [CCMR subgroup analysis design](docs/ccmr-breakdown.md)
