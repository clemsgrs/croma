#!/usr/bin/env bash
# Parametrized PathoROB-faithful Robustness Index runner.
#
# Reproduces a PathoROB-EXACT Robustness Index set for one dataset, building the
# faithful manifest from PathoROB's own selection if absent, then running benchmark.py
# with the paper's defaults. Collapses the former run_camelyon_fulldensity.sh and
# run_tolkach_reduced.sh into a single dataset-dispatched script.
#
# Usage:
#   scripts/repro/run_faithful_benchmark.sh camelyon
#   scripts/repro/run_faithful_benchmark.sh tolkach

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root: .../croma

DATASET="${1:-}"

case "$DATASET" in
camelyon)
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

  CUDA_VISIBLE_DEVICES=1 python scripts/bench/benchmark.py \
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
  ;;

tolkach)
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

  PATHOROB_META="../PathoROB/data/metadata/tolkach_esca_reduced.csv"
  MANIFEST="data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv"
  OUTDIR="output/pathorob-tolkach-esca-faithful"
  # Data-archive path (not a repo path). Override IMG_BASE to point at your own copy of
  # the PathoROB Tolkach-ESCA images; defaults to the internal archive location.
  IMG_BASE="${IMG_BASE:-/data/pathology/archives/public-datasets/pathorob/tolkach_esca/images}"
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

  python scripts/bench/benchmark.py \
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
  echo "  python scripts/repro/generate_results_table.py \\"
  echo "    --metrics $OUTDIR/results/metrics.csv \\"
  echo "    --name 'PathoROB Tolkach-ESCA' \\"
  echo "    --label tab:main-results-tolkach \\"
  echo "    --out paper/sections/results_table_tolkach.tex"
  ;;

*)
  echo "usage: $0 <camelyon|tolkach>" >&2
  exit 2
  ;;
esac
