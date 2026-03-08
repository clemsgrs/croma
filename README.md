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

ri = RI.compute(features, manifest, evaluation_design="paired_2x2", k_candidates=[5, 11, 21])
mari = MaRI.compute(features, manifest, evaluation_design="paired_2x2", k_candidates=[5, 11, 21], tau=0.2)
ccrr = CCRR.compute(
    features,
    manifest,
    evaluation_design="paired_2x2",
    m=1,
)
# optional: exclude one or more centers before RI/MaRI computation
mari_no_center_x = MaRI.compute(
    features,
    manifest,
    evaluation_design="paired_2x2",
    k_candidates=[5, 11, 21],
    tau=0.2,
    exclude_centers=["CENTER_X"],
)
```

`sample_values` contains informative evaluation-unit scores only:
- samples/occurrences with undefined per-row ratio (`SO_i + OS_i = 0`) are excluded
- `sample_values_aligned` preserves the full aligned output with `NaN` for undefined rows
- `paired_2x2` is occurrence-level and faithful to the manifest-defined PathoROB subsets
- `dataset_wide` is sample-level and evaluates the retained dataset once

CCRR neighbor search is fully automatic:
- expands neighborhood size and retries unresolved samples automatically
- targets fully defined samples by default and returns `undefined_frac`

Required manifest columns:
- `sample_id`
- `image_path`
- `label`
- `medical_center`
- `slide_id`

`evaluation_design` is required and must be one of:
- `"paired_2x2"`: explicit manifest-defined 2x2 subsets; requires a `subset` column.
- `"dataset_wide"`: single full-dataset evaluation without 2x2 decomposition.

## Why use 2x2 pairs

RI/MaRI/CCRR compare two competing neighbor signals:
- `SO`: same class, opposite center (desired)
- `OS`: opposite class, same center (undesired)

To keep this comparison meaningful and unbiased, the `"paired_2x2"` design requires each evaluation subset to be a valid **2x2 pair**:
- 2 classes
- 2 centers
- all 4 class-center cells present

The paired implementation does not infer all valid pairings from an arbitrary manifest. It only uses the explicit subset memberships provided in the manifest, matching the PathoROB contract.

## Unified Benchmark

Run feature extraction + metric computation + plot generation in one command:

```bash
python scripts/benchmark.py \
  --manifest /path/to/manifest.csv \
  --output-dir /path/to/benchmark
```

## Prepare PathoROB Data

To prepare TCGA and Tolkach ESCA from Hugging Face into local PNG tiles plus MaRI manifests:

```bash
pip install "mari[bench]"
python scripts/prepare_pathorob.py \
  --output-dir /data/pathology/projects/clement/discern/data/eval/pathorob \
  --datasets tcga,tolkach_esca \
  --revision main
```

To include Camelyon in a full refresh:

```bash
python scripts/prepare_pathorob.py \
  --output-dir /data/pathology/projects/clement/discern/data/eval/pathorob \
  --datasets camelyon,tcga,tolkach_esca \
  --revision main
```

Behavior:
- uses `--output-dir` as the shared PathoROB root
- writes images next to Camelyon under sibling dataset folders (for example `.../pathorob/tcga/images/`)
- merges multi-shard parquet datasets (for example TCGA) into one dataset output
- writes one manifest per dataset to `data/` (`pathorob-tcga.csv`, `pathorob-tolkach_esca.csv`, `pathorob-camelyon.csv`)
- for reduced paired PathoROB metadata, expands cell-level buckets into explicit runtime `subset` quartets so `paired_2x2` matches the paper design
  - binary reduced datasets use short center-pair ids such as `RUMC_UMCU`
  - multi-class reduced datasets use `LABEL1+LABEL2__CENTER1_CENTER2`
- always removes downloaded parquet payloads after conversion and validation
- writes per-dataset provenance metadata to `prepared_meta.json`

By default, all registered models are evaluated. Use `--models` to restrict to a subset.

Model-specific dependency note:
- `CONCH` requires: `pip install "git+https://github.com/Mahmoodlab/CONCH.git"`
- `CONCHv1.5` requires: `pip install "git+https://github.com/mahmoodlab/TRIDENT.git"`
With direct imports, missing benchmark dependencies fail at import/runtime immediately.

Defaults:
- `--evaluation-design paired_2x2`
- `--k-candidates 3,5,7,10,15,20,25`

Benchmark extraction note:
- when the evaluation manifest repeats the same physical sample across multiple paired subsets, embeddings are extracted once per unique `(sample_id, image_path, label, medical_center, slide_id)` source row and then reused for the repeated occurrence-level metric rows
- `--tau 0.2`
- `--ccrr-m-max 20`
- `--ccrr-alpha 0.10`
- `--progress auto` (show tqdm bars on TTY; plain logs otherwise)

Optional:
- `--continuous-k-sweep-max 100` to evaluate every integer `k` from 1 to 100 for k-sweep outputs and k-selection.
- `--exclude-center CENTER_X` (repeatable) to exclude one or more centers from evaluation.
- `--evaluation-design dataset_wide` to run one full-dataset evaluation instead of subset-defined paired evaluation.
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
- `evaluation_design`, `evaluation_unit`
- biological kNN balanced accuracy at `k` (`knn_bacc`)
- center kNN balanced accuracy at `k` (`knn_center_bacc`)
- RI at the same `k`
- MaRI at the same `k`
- selected biological `k` (`selected_k`, argmax biological `knn_bacc`; ties broken by candidate order)
- selected center `k` (`selected_k_center`, argmax center `knn_center_bacc`; ties broken by candidate order)
- excluded center signature (`excluded_centers`; comma-separated normalized names, empty when not used)

`metrics.csv`/`.json` additionally include model-level kNN summaries:
- `evaluation_design`, `evaluation_unit`
- `bio_knn_bacc`: biological balanced accuracy at `selected_k`
- `center_knn_bacc`: center balanced accuracy at `selected_k_center`
- `selected_k_center`: center task selected `k`
- `excluded_centers`: excluded center signature used for this run
- `ccrr`, `ccrr_std`, `ccrr_m`: CCRR summary (for `m=1`)
- `ccrr_undefined_frac`: undefined sample fraction for CCRR
- `ccrr_alpha`, `ccrr_q_alpha`, `ccrr_ltm_alpha`: CCRR tail statistics
- `ccrr_search`: includes CCRR search settings and alpha (`thr=...;start=...;growth=...;alpha=...`)

`ccrr_m_sweep_metrics.*` stores one row per `(model, m)` with:
- `evaluation_design`, `evaluation_unit`
- `ccrr`, `ccrr_std`
- `ccrr_undefined_frac`

`per_sample_metrics.*` stores one row per aligned evaluation unit:
- `dataset_wide`: one row per retained sample (`evaluation_unit="sample"`, `subset="dataset"`)
- `paired_2x2`: one row per retained subset occurrence (`evaluation_unit="occurrence"`, explicit `subset`)

`sample_distributions/` stores raw informative RI/MaRI/CCRR values for each model.

`metrics.csv`/`.json` also include:
- `ri_undefined_frac`, `mari_undefined_frac`: fraction of samples with undefined per-sample score (`SO_i + OS_i = 0`)

### Analyze Saved Metrics

You can compute correlations, rank tables, and rank-shift summaries from a `metrics.csv` file:

```bash
python scripts/analyze_results.py \
  --metrics-csv /path/to/results/metrics.csv \
  --rank-reference RI
```

Outputs are written to `/path/to/results/analysis/` by default:
- `correlation_pearson.csv`, `correlation_spearman.csv`
- `model_ranks.csv`
- `rank_deltas.csv`, `rank_agreement.csv`
- `top_models_by_metric.csv`
- `model_action_flags.csv` (threshold-based per-model action flags)
- `k_sweep_sensitivity.csv` (when `k_sweep_metrics.csv` is available)
- `ccrr_m_sweep_sensitivity.csv` (when `ccrr_m_sweep_metrics.csv` is available)
- `analysis_report.md`

`scripts/analyze_results.py` auto-detects sibling sweep files next to `metrics.csv` by default:
- `k_sweep_metrics.csv`
- `ccrr_m_sweep_metrics.csv`

You can also override with:
- `--k-sweep-csv /path/to/k_sweep_metrics.csv`
- `--ccrr-m-sweep-csv /path/to/ccrr_m_sweep_metrics.csv`

### Metric Cache Behavior

- Cache keys include:
  - input fingerprints (`manifest_fingerprint`, `embedding_fingerprint`, `excluded_centers`)
  - artifact-specific parameters
  - cache schema/code fingerprint
- Cache invalidation is dependency-specific:
  - changing `tau` recomputes only MaRI artifacts
  - changing CCRR search parameters recomputes only CCRR artifacts
  - changing `k` candidates recomputes kNN/RI/MaRI artifacts only
  - changing `evaluation_design` recomputes kNN/RI/MaRI/CCRR artifacts
- Report files (`metrics.csv`, `k_sweep_metrics.csv`, `ccrr_m_sweep_metrics.csv`) are rewritten every run from current artifacts.
- `--recompute-metrics` bypasses cache reads but still refreshes cache artifacts.
