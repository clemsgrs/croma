# MaRI

Lightweight Python package for:
- RI (Robustness Index)
- MaRI (Margin-aware Robustness Index)
- CCRR (Cross-Confounder Retrieval Ratio)

## Install

```bash
pip install mari
```

## Quickstart

```python
import numpy as np
import pandas as pd
from mari import CCRR, RI, MaRI

manifest = pd.read_csv("data/prostate-shift-binary.csv")
features = np.load("embeddings.npy")  # shape: (N, D)

ri = RI.compute(features, manifest, mode="paired", k_candidates=[5, 11, 21])
mari = MaRI.compute(features, manifest, mode="paired", k_candidates=[5, 11, 21], tau=0.2)
ccrr = CCRR.compute(
    features,
    manifest,
    mode="paired",
    m=1,
)
# optional: exclude one or more centers before RI/MaRI computation
mari_no_center_x = MaRI.compute(
    features,
    manifest,
    mode="paired",
    k_candidates=[5, 11, 21],
    tau=0.2,
    exclude_centers=["CENTER_X"],
)
```

`sample_values` contains informative per-sample scores only:
- samples with undefined per-sample ratio (`SO_i + OS_i = 0`) are excluded
- in `mode="paired"`, repeated appearances of the same sample across valid 2x2 pairs are averaged to one value per sample

CCRR neighbor search is fully automatic:
- expands neighborhood size and retries unresolved samples automatically
- targets fully defined samples by default and returns `undefined_frac`

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

RI/MaRI/CCRR compare two competing neighbor signals:
- `SO`: same class, opposite center (desired)
- `OS`: opposite class, same center (undesired)

To keep this comparison meaningful and unbiased, the `"paired"` mode builds each evaluation subset as a valid **2x2 pair**:
- 2 classes
- 2 centers
- all 4 class-center cells present

This avoids one-vs-rest base-rate bias (where large complements dominate neighbors), keeping the metric interpretable.

## Unified Benchmark

Run feature extraction + metric computation + plot generation in one command:

```bash
python scripts/benchmark.py \
  --manifest /path/to/manifest.csv \
  --output-dir /path/to/benchmark
```

By default, all registered models are evaluated. Use `--models` to restrict to a subset.

Model-specific dependency note:
- `CONCH` requires: `pip install "git+https://github.com/Mahmoodlab/CONCH.git"`
- `CONCHv1.5` requires: `pip install "git+https://github.com/mahmoodlab/TRIDENT.git"`

These dependencies are validated in the embedding loader when the corresponding model is used.

Defaults:
- `--mode global`
- `--k-candidates 3,5,7,10,15,20,25`
- `--tau 0.2`
- `--ccrr-m-candidates 1,5,10,15,20`
- `--ccrr-alpha 0.10`
- `--progress auto` (show tqdm bars on TTY; plain logs otherwise)

Optional:
- `--continuous-k-sweep-max 100` to evaluate every integer `k` from 1 to 100 for k-sweep outputs and k-selection.
- `--exclude-center CENTER_X` (repeatable) to exclude one or more centers from evaluation.
- `--ccrr-m-candidates ...` to override CCRR sweep values (must include `1`).
- `--ccrr-alpha ...` to choose the CCRR tail percentile used for `Q_alpha` and `LTM_alpha`.
- `--recompute-metrics` to bypass metric cache reads and force full metric recomputation.
- `--progress off` to force log-only output (recommended for CI/log redirects).

Outputs:
- all artifacts are stored under a dataset folder: `<output-dir>/<manifest_stem>/`
- embeddings: `<output-dir>/<manifest_stem>/embeddings/<model>.npy` (+ `.json`)
- metrics:
  - `<output-dir>/<manifest_stem>/results/metrics.csv`
  - `<output-dir>/<manifest_stem>/results/metrics.json`
  - `<output-dir>/<manifest_stem>/results/k_sweep_metrics.csv`
  - `<output-dir>/<manifest_stem>/results/k_sweep_metrics.json`
  - `<output-dir>/<manifest_stem>/results/ccrr_m_sweep_metrics.csv`
  - `<output-dir>/<manifest_stem>/results/ccrr_m_sweep_metrics.json`
  - `<output-dir>/<manifest_stem>/results/sample_distributions/ri.<model>.npy` (+ `.json`)
  - `<output-dir>/<manifest_stem>/results/sample_distributions/mari.<model>.npy` (+ `.json`)
  - `<output-dir>/<manifest_stem>/results/sample_distributions/ccrr.<model>.npy`
  - `<output-dir>/<manifest_stem>/results/cache/index.jsonl`
  - `<output-dir>/<manifest_stem>/results/cache/artifacts/<artifact>/<model>/<key_hash>.json|.npy`
- plots:
  - `<output-dir>/<manifest_stem>/plots/knn_bio_k_sweep.png`
  - `<output-dir>/<manifest_stem>/plots/knn_center_k_sweep.png`
  - `<output-dir>/<manifest_stem>/plots/ri_k_sweep.png`
  - `<output-dir>/<manifest_stem>/plots/mari_k_sweep.png`
  - `<output-dir>/<manifest_stem>/plots/ccrr_m_sweep.png`
  - `<output-dir>/<manifest_stem>/plots/ccrr_ltm_comparison.png`
  - `<output-dir>/<manifest_stem>/plots/bio_vs_center_scatter.png`
  - `<output-dir>/<manifest_stem>/plots/mari_vs_ri_scatter.png`
  - `<output-dir>/<manifest_stem>/plots/ccrr_vs_mari_scatter.png`
  - `<output-dir>/<manifest_stem>/plots/ccrr_sample_distributions.png`
  - `<output-dir>/<manifest_stem>/plots/benchmark_6panel_summary.png`

`k_sweep_metrics.*` stores one row per `(model, k)` with:
- biological kNN balanced accuracy at `k` (`knn_bacc`)
- center kNN balanced accuracy at `k` (`knn_center_bacc`)
- RI at the same `k`
- MaRI at the same `k`
- selected biological `k` (`selected_k`, argmax biological `knn_bacc`; ties broken by candidate order)
- selected center `k` (`selected_k_center`, argmax center `knn_center_bacc`; ties broken by candidate order)
- excluded center signature (`excluded_centers`; comma-separated normalized names, empty when not used)

`metrics.csv`/`.json` additionally include model-level kNN summaries:
- `bio_knn_bacc`: biological balanced accuracy at `selected_k`
- `center_knn_bacc`: center balanced accuracy at `selected_k_center`
- `selected_k_center`: center task selected `k`
- `excluded_centers`: excluded center signature used for this run
- `ccrr`, `ccrr_std`, `ccrr_m`: CCRR summary (for `m=1`)
- `ccrr_undefined_frac`: undefined sample fraction for CCRR
- `ccrr_alpha`, `ccrr_q_alpha`, `ccrr_ltm_alpha`: CCRR tail statistics
- `ccrr_search`: includes CCRR search settings and alpha (`thr=...;start=...;growth=...;alpha=...`)

`ccrr_m_sweep_metrics.*` stores one row per `(model, m)` with:
- `ccrr`, `ccrr_std`
- `ccrr_undefined_frac`

`sample_distributions/` stores raw per-sample RI/MaRI/CCRR values for each model.

`metrics.csv`/`.json` also include:
- `ri_undefined_frac`, `mari_undefined_frac`: fraction of samples with undefined per-sample score (`SO_i + OS_i = 0`)

### Metric Cache Behavior

- Cache keys include:
  - input fingerprints (`manifest_fingerprint`, `embedding_fingerprint`, `excluded_centers`)
  - artifact-specific parameters
  - cache schema/code fingerprint
- Cache invalidation is dependency-specific:
  - changing `tau` recomputes only MaRI artifacts
  - changing CCRR search parameters recomputes only CCRR artifacts
  - changing `k` candidates recomputes kNN/RI/MaRI artifacts only
  - changing `mode` recomputes RI/MaRI/CCRR artifacts (kNN artifacts are reused)
- Report files (`metrics.csv`, `k_sweep_metrics.csv`, `ccrr_m_sweep_metrics.csv`) are rewritten every run from current artifacts.
- `--recompute-metrics` bypasses cache reads but still refreshes cache artifacts.
