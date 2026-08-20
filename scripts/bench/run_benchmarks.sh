#!/usr/bin/env bash
# Compute metrics for one protocol across registered benchmarks, then render each run.
#
#   scripts/repro/run_benchmarks.sh k-star                 # every benchmark
#   scripts/repro/run_benchmarks.sh median-k pathorob-camelyon   # just one
#
# CROMA_K_GRID=sparse sweeps PathoROB's k grid instead of every integer 1..k_max.
#
# This replaces run_median_k.sh. Everything that driver did by hand -- copying each
# benchmark's manifest, symlinking an embeddings directory next to it, and pasting a
# model roster and k-max per benchmark -- is now read from the benchmark registry
# (scripts/bench/benchmarks.py). Embeddings are never touched: benchmark.py is pure-read
# and gathers the rows its benchmark defines out of the shared tileset matrix.
set -uo pipefail
cd "$(dirname "$0")/../.."

PROTOCOL="${1:-}"
if [ -z "$PROTOCOL" ]; then
  echo "usage: $0 <k-star|median-k> [benchmark ...]" >&2
  exit 2
fi
shift

OMP_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="src:scripts/bench"

# The cd above puts us at the repo root, which is the only root there is: `output/` here
# is the same tree scripts/bench/layout.py resolves.
OUT_ROOT="output"

# The k grid is part of the protocol (it bounds which k a model can select), so it is
# recorded in each run's `k_values` column rather than being a display setting.
K_GRID_ARGS=()
if [ -n "${CROMA_K_GRID:-}" ]; then
  K_GRID_ARGS=(--k-grid "$CROMA_K_GRID")
fi

if [ "$#" -gt 0 ]; then
  BENCHMARKS=("$@")
else
  mapfile -t BENCHMARKS < <(python -c "import benchmarks; print('\n'.join(benchmarks.BENCHMARKS))")
fi

failed=()
for B in "${BENCHMARKS[@]}"; do
  echo "=================================================================="
  echo ">>> $B  (protocol=$PROTOCOL)  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$OMP_THREADS" \
    python scripts/bench/benchmark.py --benchmark "$B" --protocol "$PROTOCOL" \
      ${K_GRID_ARGS[@]+"${K_GRID_ARGS[@]}"}
  rc=$?
  M="$OUT_ROOT/metrics/$PROTOCOL/$B/results/metrics.csv"
  if [ $rc -eq 0 ] && [ -f "$M" ]; then
    python -c "
import pandas as pd
d = pd.read_csv('$M')
print('  OK  k (unique):', sorted(d['k'].unique()),
      '| mean support=%.1f%%' % (100 * d['support'].mean()),
      '| n_models=%d' % len(d))
"
    python scripts/bench/render.py "$OUT_ROOT/metrics/$PROTOCOL/$B"
  else
    echo "  !! FAILED rc=$rc (metrics missing: $M)"
    failed+=("$B")
  fi
done

echo "=================================================================="
if [ ${#failed[@]} -gt 0 ]; then
  echo ">>> FAILED: ${failed[*]}"
  exit 1
fi
echo ">>> DONE ($PROTOCOL, ${#BENCHMARKS[@]} benchmarks) $(date +%H:%M:%S)"
