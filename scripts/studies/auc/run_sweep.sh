#!/bin/bash
# Run every missing or stale AUROC cell, WORKERS at a time (default 3).
#
# The work list comes from auc_cells.py: one cell per APD-study cell, less the ones that
# exist and still reproduce their APD cell, the slowest cohort first. Each cell publishes
# its own JSON under output/studies/auc/ atomically, so this is safe to re-launch after a
# preemption: it resumes where the last run stopped, and re-runs any cell that an APD
# re-run has made stale.
#
# Model names contain spaces ("RudolfV 2-B"), so xargs splits on newlines only and each
# worker splits its tab-separated line itself.
#
#   scripts/studies/auc/run_sweep.sh [WORKERS] >> output/studies/auc/sweep.log 2>&1
#   python scripts/studies/auc/auc_cells.py check     # exits 0 once every cell is current
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS="${1:-3}"
PYTHON="${PYTHON:-python3}"
export HERE PYTHON
# One BLAS thread per process: the probe grid search parallelises poorly inside a process,
# so the driver fans out across cells instead.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

pending="$("$PYTHON" "$HERE/auc_cells.py" pending)" || exit 1
echo "$(date -Is) start: $(printf '%s' "$pending" | grep -c .) cell(s), $WORKERS worker(s)"
status=0
if [ -n "$pending" ]; then
  printf '%s\n' "$pending" | xargs -d '\n' -P "$WORKERS" -n 1 bash -c '
    IFS=$'"'"'\t'"'"' read -r dataset model <<< "$0"
    "$PYTHON" "$HERE/probe_auc_cell.py" --dataset "$dataset" --model "$model"
  ' || status=$?
fi
"$PYTHON" "$HERE/auc_cells.py" check | tail -n 1
echo "$(date -Is) ALL DONE (xargs exit $status)"
exit "$status"
