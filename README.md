# MaRI

Lightweight Python package for:
- RI (Robustness Index)
- MaRI (Margin-aware Robustness Index)
- Tail summaries (percentiles and lower-tail mean)

## Install

```bash
pip install mari
```

## Quickstart

```python
import numpy as np
import pandas as pd
from mari import RI, MaRI, tail_percentile, lower_tail_mean

manifest = pd.read_csv("data/prostate-shift-binary.csv")
features = np.load("embeddings.npy")  # shape: (N, D)

ri = RI.compute(features, manifest, mode="paired", k_candidates=[5, 11, 21])
mari = MaRI.compute(features, manifest, mode="paired", k_candidates=[5, 11, 21], tau=0.2)

q10 = tail_percentile(mari.sample_values, 10.0)
ltm10 = lower_tail_mean(mari.sample_values, 10.0)
```

Required manifest columns:
- `sample_id`
- `image_path`
- `label`
- `medical_center`
- `slide_id`

`mode` is required and must be one of:
- `"paired"`: 2x2 pairing and aggregation.
- `"global"`: single full-dataset evaluation without 2x2 decomposition.

## Why use 2x2 pairs

RI/MaRI compare two competing neighbor signals:
- `SO`: same class, opposite center (desired)
- `OS`: opposite class, same center (undesired)

To keep this comparison meaningful and unbiased, the `"paired"` mode builds each evaluation subset as a valid **2x2 pair**:
- 2 classes
- 2 centers
- all 4 class-center cells present

This avoids one-vs-rest base-rate bias (where large complements dominate neighbors), keeping the metric interpretable.

## Publishing

PyPI publishing is automated on GitHub Releases:
- workflow: `/Volumes/temporary/clement/code/MaRI/.github/workflows/release.yaml`
- docs: `/Volumes/temporary/clement/code/MaRI/docs/releasing.md`
- helper script: `/Volumes/temporary/clement/code/MaRI/release.py`
