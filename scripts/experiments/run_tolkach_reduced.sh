#!/usr/bin/env bash
# Run the PathoROB-EXACT Tolkach-ESCA Robustness Index set.
#
# Verified against the PathoROB source (../PathoROB): get_meta() loads the
# `tolkach_esca_reduced` metadata for Tolkach and computes RI NON-paired
# (paired evaluation is implemented for TCGA only; it raises for Tolkach/Camelyon),
# with the OOD subset excluded. That canonical set is:
#   3 centres (VALSET1_UKK / VALSET2_WNS / VALSET4_CHA_FULL; TCGA dropped),
#   6 tissue classes, 500 patches per (class x centre) = 9,000 patches, 43 slides.
#   ("5 cases/combination x 100 patches/case, excluding TCGA", paper Sup. Note B.3.)
#
# The manifest below is built to reproduce PathoROB's selection EXACTLY: every row
# is joined from ../PathoROB/data/metadata/tolkach_esca_reduced.csv via
# sample_id = "<slide_id>__<patch_id>" (0 missing images; 9000/9000 match).
#
# NOTE: this supersedes an earlier (incorrect) prep that used our 90k *-reduced
# manifest with --evaluation-design paired_2x2. That was wrong twice over:
# PathoROB uses 9k (not 90k) and dataset_wide / non-paired (not paired_2x2).
#
# This becomes the MAIN Tolkach results table. The 4-centre `output/pathorob-tolkach-esca`
# run (TCGA included) is kept ONLY for the pretraining-overlap test (tab:pretraining-overlap).
#
# Hyperparameters pinned to the rest of the paper (tau=0.2, k_max=25, m<=20,
# CRoMa search start=200/growth=2/alpha=0.1 -- all benchmark.py defaults).

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root: .../croma

PATHOROB_META="../PathoROB/data/metadata/tolkach_esca_reduced.csv"
MANIFEST="data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv"
OUTDIR="output/pathorob-tolkach-esca-faithful"
IMG_BASE="/data/pathology/archives/public-datasets/pathorob/tolkach_esca/images"
MODELS="CONCH,CONCHv1.5,H-optimus-0,H-optimus-1,H0-mini,Hibou-B,Hibou-L,Midnight-12k,Phikon,Phikon-v2,Prost40M,Prov-GigaPath,UNI,UNI2-h,Virchow,Virchow2"

# Build the exact 9k manifest from PathoROB's metadata if absent.
if [ ! -f "$MANIFEST" ]; then
  echo "Building $MANIFEST from $PATHOROB_META (exact PathoROB selection)..."
  python - "$PATHOROB_META" "$MANIFEST" "$IMG_BASE" <<'PY'
import csv, os, sys
src, dst, base = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(src)))
out, miss = [], 0
for r in rows:
    sid = f"{r['slide_id']}__{r['patch_id']}"
    img = f"{base}/{sid}.png"
    miss += (not os.path.exists(img))
    out.append({"sample_id": sid, "image_path": img, "label": r["biological_class"],
                "medical_center": r["medical_center"], "slide_id": r["slide_id"]})
assert miss == 0, f"{miss} missing images"
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "label", "medical_center", "slide_id"])
    w.writeheader(); w.writerows(out)
print(f"wrote {dst}: {len(out)} patches, {len(set(r['slide_id'] for r in out))} slides")
PY
fi

python scripts/benchmark.py \
  --manifest "$MANIFEST" \
  --confounder-column medical_center \
  --evaluation-design dataset_wide \
  --models "$MODELS" \
  --tau 0.2 \
  --k-max 25 \
  --output-dir "$OUTDIR" \
  --device auto

echo
echo "Done. Metrics: $OUTDIR/results/metrics.csv"
echo "Next: regenerate the main Tolkach table from this run, e.g."
echo "  python scripts/experiments/generate_results_table.py \\"
echo "    --metrics $OUTDIR/results/metrics.csv \\"
echo "    --name 'PathoROB Tolkach-ESCA' \\"
echo "    --label tab:main-results-tolkach \\"
echo "    --out paper/sections/results_table_tolkach.tex"
