#!/usr/bin/env bash
# Median-of-k* regeneration for ALL SIX benchmarks in one pass.
#
# Merges the former two batch drivers (run_median_k_batch.sh + run_median_k_batch2.sh)
# into a single parametrised loop: the four main benchmarks (PANDA-ISUP, Tolkach,
# Camelyon, TCGA-2x2) plus the two tractable supp dataset-wide benchmarks
# (prostate-shift, TCGA-4x4). PANDA-ISUP (paired_2x2) elsewhere stays at k*; here every
# benchmark listed below is regenerated at the shared median-of-k* operating point.
#
# Re-runs benchmark.py --use-median-k over the ALREADY-CACHED embeddings (no GPU /
# no re-extraction): each benchmark's own embedding_source_manifest.csv is the exact
# eval input the embeddings were extracted from, so passing it back guarantees an
# embedding-cache hit AND byte-identical eval data -- only the k operating point moves
# from per-model k* to the shared median-of-k*.
#
# The faithful/* (k*) dirs are left UNTOUCHED as the swap-back backup; median outputs
# land under output/faithful/median/<name>-median/. $ROOT is cleaned once at the start,
# then all six benchmarks are built up under it in a single loop.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
ROOT="output/faithful/median"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
rm -rf "$ROOT"; mkdir -p "$ROOT/manifests"

# Model rosters: the tile benchmarks share the 16 tile models; PANDA uses 4 slide models.
TILE16="CONCH,CONCHv1.5,H-optimus-0,H-optimus-1,H0-mini,Hibou-B,Hibou-L,Midnight-12k,Phikon,Phikon-v2,Prost40M,Prov-GigaPath,UNI,UNI2-h,Virchow,Virchow2"
PANDA4="PRISM,TITAN,Prov-GigaPath,MOOZY"

# name | source dir (embeddings + embedding_source_manifest) | design | k-max | models
BENCHES=(
  "panda-median|output/panda-wsi-cancer|dataset_wide|25|$PANDA4"
  "tolkach-median|output/faithful/k-star/pathorob-tolkach-esca-faithful|dataset_wide|100|$TILE16"
  "camelyon-median|output/faithful/k-star/pathorob-camelyon-faithful|dataset_wide|100|$TILE16"
  "tcga2x2-median|output/faithful/k-star/pathorob-tcga-2x2|paired_2x2|100|$TILE16"
  "prostate-median|output/prostate-shift-binary-kirumc|dataset_wide|25|$TILE16"
  "tcga4x4-median|output/faithful/k-star/pathorob-tcga-4x4|dataset_wide|100|$TILE16"
)

for spec in "${BENCHES[@]}"; do
  IFS='|' read -r NAME SRC DESIGN KMAX MODELS <<< "$spec"
  echo "=================================================================="
  echo ">>> $NAME  (src=$SRC  design=$DESIGN  k-max=$KMAX)  $(date +%H:%M:%S)"
  cp "$SRC/embedding_source_manifest.csv" "$ROOT/manifests/$NAME.csv"
  mkdir -p "$ROOT/$NAME"
  ln -sfn "$REPO/$SRC/embeddings" "$ROOT/$NAME/embeddings"
  # Step 1: compute metrics (no plotting).
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$OMP_THREADS" PYTHONPATH=src python scripts/bench/benchmark.py \
    --manifest "$ROOT/manifests/$NAME.csv" \
    --confounder-column confounder \
    --evaluation-design "$DESIGN" \
    --models "$MODELS" \
    --k-max "$KMAX" --use-median-k --device cpu \
    --output-dir "$ROOT"
  rc=$?
  M="$ROOT/$NAME/results/metrics.csv"
  if [ $rc -eq 0 ] && [ -f "$M" ]; then
    python3 -c "
import pandas as pd
d=pd.read_csv('$M')
d['support']=(1-d['ri_undefined_frac'])*100
print('  OK  median-k (unique):', sorted(d['k'].unique()),
      '| mean support=%.1f%%'%d['support'].mean(),
      '| n_models=%d'%len(d))
"
    # Step 2: render this run's figure set from the written metrics.
    PYTHONPATH=src python scripts/bench/render.py "$ROOT/$NAME"
  else
    echo "  !! FAILED rc=$rc (metrics missing: $M)"
  fi
done
echo "=================================================================="
echo ">>> ALL SIX DONE $(date +%H:%M:%S)"
