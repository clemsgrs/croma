#!/usr/bin/env bash
# Confirmatory full-density Camelyon run — PathoROB's exact Camelyon RI set.
#
# Why this run: our main Camelyon table uses a ~5x-thinned subsample (4,000 patches,
# output/pathorob-camelyon-reduced). PathoROB instead computes RI on the FULL in-domain
# Camelyon set (Sup. Note B.3 / Table 3): RUMC + UMCU only, 300 patches/slide, 68 slides,
# 20,400 patches total, with the 3 OOD centres (CWZ/RST/LPON) excluded.
#
# Camelyon is a native 2x2 (normal/tumor x RUMC/UMCU), so PathoROB applies RI directly:
# dataset_wide over the two ID centres IS the PathoROB protocol -- no paired-quartet
# construction is needed (that is only for many-class datasets like TCGA/Tolkach).
#
# Purpose: verify the low RI/MaRI *support* on Camelyon (the coverage-hazard headline,
# 8-21% in results_table.tex) is intrinsic geometry, not a 4,000-patch subsampling
# artifact. CRoMa is k-free / fully covered and is unaffected either way.
#
# Hyperparameters pinned to the rest of the paper (tau=0.2, k_max=25, m<=20,
# CRoMa search start=200/growth=2/alpha=0.1 -- all benchmark.py defaults).

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root: .../croma

SRC="data/pathorob/manifests/pathorob-camelyon.csv"           # 5 centres, has id_ood flag
MANIFEST="data/pathorob/manifests/pathorob-camelyon-faithful.csv"  # RUMC+UMCU only (PathoROB ID set)
OUTDIR="output"

# Build the PathoROB-faithful manifest (the two in-domain centres) if absent.
if [ ! -f "$MANIFEST" ]; then
  echo "Generating $MANIFEST (RUMC + UMCU, the 20,400-patch PathoROB RI set)..."
  python - "$SRC" "$MANIFEST" <<'PY'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    rows = list(csv.DictReader(f)); cols = rows[0].keys()
keep = [r for r in rows if r["medical_center"] in ("RUMC", "UMCU")]
assert len(keep) == 20400, f"expected 20400 in-domain patches, got {len(keep)}"
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cols)); w.writeheader(); w.writerows(keep)
print(f"wrote {dst}: {len(keep)} patches, "
      f"{len(set(r['slide_id'] for r in keep))} slides")
PY
fi

CUDA_VISIBLE_DEVICES=1 python scripts/benchmark.py \
  --manifest "$MANIFEST" \
  --confounder-column medical_center \
  --evaluation-design dataset_wide \
  --output-dir "$OUTDIR"

echo
echo "Done. Metrics: $OUTDIR/results/metrics.csv"
echo "Compare support (1 - ri_undefined_frac) vs the 4k subsample:"
echo "  python - <<'PY'"
echo "  import pandas as pd"
echo "  a=pd.read_csv('output/pathorob-camelyon-reduced-kstar/results/metrics.csv').set_index('model')"
echo "  b=pd.read_csv('$OUTDIR/results/metrics.csv').set_index('model')"
echo "  s=lambda d:(1-d['ri_undefined_frac'])*100"
echo "  print(pd.DataFrame({'support_4k':s(a).round(1),'support_20k':s(b).round(1)}).sort_values('support_20k'))"
echo "  PY"
