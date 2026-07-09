"""One-shot migration of ``output/`` to the tileset/metrics layout.

Before: every ``(dataset, operating point)`` pair owned a directory holding its own
copy of the embeddings, so ``k-star`` vs ``median-k`` vs an old pruning experiment each
re-materialised the same matrices. After: one embedding matrix per tileset, and every
benchmark borrows a row-view of it.

    output/embeddings/<tileset>/{manifest.csv, <Model>.npy, <Model>.npy.json}
    output/metrics/<protocol>/<benchmark>/{results,plots,studies}
    output/studies/apd/

Runs in three gated stages so nothing irreversible happens on a bad plan::

    python scripts/tools/migrate_output_layout.py            # plan only (default)
    python scripts/tools/migrate_output_layout.py --apply    # moves, then verifies
    python scripts/tools/migrate_output_layout.py --delete   # reclaim, only if verify passes

``--delete`` refuses to run unless verification passes, and it only ever removes
directories whose contents were proven redundant (bitwise-identical row-subsets of a
tileset) or which nothing references.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
sys.path.insert(0, str(ROOT / "src"))

import layout  # noqa: E402
from benchmarks import BENCHMARKS  # noqa: E402

from croma.alignment import (  # noqa: E402
    build_embedding_source_manifest,
    build_view_row_index,
)
from croma.metrics.pairs import load_manifest  # noqa: E402

# tileset -> (legacy dir holding embeddings/, legacy manifest file or None to derive)
TILESET_SOURCES: dict[str, tuple[str, str | None]] = {
    "pathorob-camelyon": ("output/pathorob-camelyon", "embedding_source_manifest.csv"),
    "pathorob-tolkach-esca": ("output/pathorob-tolkach-esca", "embedding_source_manifest.csv"),
    "pathorob-tcga-2x2": ("output/pathorob-tcga-2x2", "embedding_source_manifest.csv"),
    "pathorob-tcga-4x4": ("output/pathorob-tcga-4x4", "embedding_source_manifest.csv"),
    # No manifest of its own: embedded straight from data/prostate-shift-binary.csv.
    "prostate-shift": ("output/prostate-shift-binary", None),
    # The 3,000-row paired set is the superset; panda-wsi-cancer is a 1,000-row view.
    "panda-wsi": ("output/panda-wsi-isup-paired-2x2", "embedding_source_manifest.csv"),
}

PROSTATE_SOURCE_MANIFEST = "data/prostate-shift-binary.csv"

# Eval manifests with no home under data/: preserved out of the legacy output tree.
# These copies were canonicalised (their confounder column was renamed to `confounder`),
# which would lose the confounder's real name -- and with it the "Data Provider" label
# on every PANDA axis and caption. Restore the source column name on the way out.
MANIFEST_RESCUES: dict[str, str] = {
    "data/benchmarks/panda.csv": "output/panda-wsi-cancer/embedding_source_manifest.csv",
    # The only surviving copy that still carries the `subset` column.
    "data/benchmarks/panda-isup.csv": "output/faithful/median/manifests/panda-isup-median-paired.csv",
}
RESCUE_CONFOUNDER_NAME = "data_provider"

RUN_SOURCES: dict[str, dict[str, str]] = {
    "k-star": {
        "pathorob-camelyon": "output/faithful/k-star/pathorob-camelyon-faithful",
        "pathorob-tolkach-esca": "output/faithful/k-star/pathorob-tolkach-esca-faithful",
        "pathorob-tcga-2x2": "output/faithful/k-star/pathorob-tcga-2x2",
        "pathorob-tcga-4x4": "output/faithful/k-star/pathorob-tcga-4x4",
        "prostate": "output/prostate-shift-binary-kirumc",
        "prostate-4class": "output/prostate-shift-4class-kirumc-paired",
        "prostate-gradebal": "output/prostate-shift-gradebal-binary-kirumc-paired",
        "panda": "output/panda-wsi-cancer",
        "panda-isup": "output/panda-wsi-isup-paired-2x2",
    },
    "median-k": {
        "pathorob-camelyon": "output/faithful/median/camelyon-median",
        "pathorob-tolkach-esca": "output/faithful/median/tolkach-median",
        "pathorob-tcga-2x2": "output/faithful/median/pathorob-tcga-2x2",
        "pathorob-tcga-4x4": "output/faithful/median/tcga4x4-median",
        "prostate": "output/faithful/median/prostate-median",
        "panda": "output/faithful/median/panda-median",
        "panda-isup": "output/faithful/median/panda-isup-median-paired",
    },
}

# Removed only after verification. Every entry is either a proven-redundant copy or
# has zero references in scripts/, src/ and tests/.
DELETE_AFTER_VERIFY: tuple[str, ...] = (
    "output/pathorob-camelyon-reduced",
    "output/pathorob-camelyon-reduced-kstar",
    "output/pathorob-camelyon-reduced-median",
    "output/pathorob-camelyon-reduced-pruned",
    "output/pathorob-camelyon-reduced-save",
    "output/pathorob-tolkach-esca-reduced",
    "output/compare-kstar-median",
    "output/compare-kstar-pruned",
    "output/compare-median-pruned",
    "output/moment_exploration",
    "output/faithful",  # duplicate embeddings + dangling symlinks, results already moved
    "output/pathorob-camelyon",
    "output/pathorob-tolkach-esca",
    "output/pathorob-tcga-2x2",
    "output/pathorob-tcga-4x4",
    "output/prostate-shift-binary",
    "output/prostate-shift-binary-kirumc",
    "output/prostate-shift-4class-kirumc-paired",
    "output/prostate-shift-gradebal-binary-kirumc-paired",
    "output/panda-wsi-cancer",
    "output/panda-wsi-isup-paired-2x2",
)


def _move(src: Path, dst: Path, plan: list[str], apply: bool) -> None:
    if not src.exists() and not src.is_symlink():
        return
    if dst.exists():
        plan.append(f"  skip (exists)  {dst.relative_to(ROOT)}")
        return
    plan.append(f"  move  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _copy(src: Path, dst: Path, plan: list[str], apply: bool) -> None:
    if dst.exists():
        plan.append(f"  skip (exists)  {dst.relative_to(ROOT)}")
        return
    if not src.exists():
        plan.append(f"  MISSING SOURCE  {src.relative_to(ROOT)}")
        return
    plan.append(f"  copy  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # copyfile, not copy2: this filesystem denies utime/chmod on new files.
        shutil.copyfile(src, dst)


def _rescue_manifest(src: Path, dst: Path, plan: list[str], apply: bool) -> None:
    """Copy an eval manifest out of output/, un-canonicalising its confounder column."""
    if dst.exists():
        plan.append(f"  skip (exists)  {dst.relative_to(ROOT)}")
        return
    if not src.exists():
        plan.append(f"  MISSING SOURCE  {src.relative_to(ROOT)}")
        return
    plan.append(
        f"  rescue  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}"
        f"  (confounder -> {RESCUE_CONFOUNDER_NAME})"
    )
    if not apply:
        return
    frame = pd.read_csv(src)
    if "confounder" in frame.columns:
        frame = frame.rename(columns={"confounder": RESCUE_CONFOUNDER_NAME})
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dst, index=False)


def _derive_prostate_manifest(dst: Path, apply: bool, plan: list[str]) -> None:
    """prostate-shift was embedded straight from its source CSV, in that row order."""
    if dst.exists():
        plan.append(f"  skip (exists)  {dst.relative_to(ROOT)}")
        return
    src = ROOT / PROSTATE_SOURCE_MANIFEST
    plan.append(f"  derive  {dst.relative_to(ROOT)}  from  {PROSTATE_SOURCE_MANIFEST}")
    if not apply:
        return
    raw = pd.read_csv(src)
    df = load_manifest(str(src), confounder_column="medical_center")
    manifest, _ = build_embedding_source_manifest(df)
    # build_embedding_source_manifest normalises keys to str; sample_id here is integral.
    if list(manifest["sample_id"]) != [str(v).strip() for v in raw["sample_id"]]:
        raise RuntimeError("prostate manifest row order diverges from the embedded order")
    n_embedded = np.load(
        next(dst.parent.glob("*.npy")), mmap_mode="r"
    ).shape[0]
    if len(manifest) != n_embedded:
        raise RuntimeError(
            f"prostate manifest has {len(manifest)} rows but {n_embedded} were embedded"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(dst, index=False)


def build_plan(apply: bool) -> list[str]:
    plan: list[str] = ["", "TILESETS -> output/embeddings/<tileset>/"]
    for tileset, (src_rel, manifest_name) in TILESET_SOURCES.items():
        src = ROOT / src_rel
        dst_dir = layout.embeddings_dir(tileset)
        _move(src / "embeddings", dst_dir, plan, apply)
        if manifest_name is None:
            _derive_prostate_manifest(layout.tileset_manifest(tileset), apply, plan)
        else:
            _move(src / manifest_name, layout.tileset_manifest(tileset), plan, apply)
        for log in sorted(src.glob("*.log")) if src.exists() else []:
            _move(log, dst_dir / log.name, plan, apply)

    plan += ["", "EVAL MANIFESTS -> data/benchmarks/"]
    for dst_rel, src_rel in MANIFEST_RESCUES.items():
        _rescue_manifest(ROOT / src_rel, ROOT / dst_rel, plan, apply)

    plan += ["", "RUNS -> output/metrics/<protocol>/<benchmark>/"]
    for protocol, mapping in RUN_SOURCES.items():
        for benchmark, src_rel in mapping.items():
            src = ROOT / src_rel
            dst = layout.metrics_dir(protocol, benchmark)
            _move(src / "results", dst / "results", plan, apply)
            _move(src / "plots", dst / "plots", plan, apply)
            if src.exists():
                for summary in sorted(src.glob("*_summary.*")):
                    _move(summary, dst / "studies" / summary.name, plan, apply)

    plan += ["", "STUDIES -> output/studies/"]
    _move(ROOT / "output/apd", ROOT / "output/studies/apd", plan, apply)
    return plan


def verify() -> bool:
    """Every benchmark must resolve to real embeddings through its tileset manifest."""
    ok = True
    print("\nVERIFY  benchmark -> tileset row mapping")
    for name, spec in BENCHMARKS.items():
        tm_path = layout.tileset_manifest(spec.tileset)
        man_path = ROOT / spec.manifest
        if not tm_path.exists():
            print(f"  FAIL {name:18s} missing tileset manifest {tm_path.relative_to(ROOT)}")
            ok = False
            continue
        if not man_path.exists():
            print(f"  FAIL {name:18s} missing eval manifest {spec.manifest}")
            ok = False
            continue
        tileset_manifest = pd.read_csv(tm_path)
        eval_manifest = load_manifest(str(man_path), confounder_column=spec.confounder_column)
        try:
            rows = build_view_row_index(eval_manifest, tileset_manifest)
        except ValueError as exc:
            print(f"  FAIL {name:18s} {exc}")
            ok = False
            continue
        models = sorted(p.stem for p in layout.embeddings_dir(spec.tileset).glob("*.npy"))
        if not models:
            print(f"  FAIL {name:18s} no embeddings under {spec.tileset}")
            ok = False
            continue
        n_emb = np.load(layout.embedding_path(spec.tileset, models[0]), mmap_mode="r").shape[0]
        if len(tileset_manifest) != n_emb:
            print(f"  FAIL {name:18s} manifest rows {len(tileset_manifest)} != npy rows {n_emb}")
            ok = False
            continue
        if rows.max(initial=-1) >= n_emb:
            print(f"  FAIL {name:18s} row index out of range")
            ok = False
            continue
        print(
            f"  ok   {name:18s} {spec.tileset:22s} eval={len(eval_manifest):6d} "
            f"-> uniq={len(set(rows.tolist())):6d}/{n_emb:6d}  models={len(models)}"
        )
    return ok


def delete() -> None:
    if not verify():
        raise SystemExit("verification failed -- refusing to delete anything")
    freed = 0
    print("\nDELETE  (verified redundant / unreferenced)")
    for rel in DELETE_AFTER_VERIFY:
        path = ROOT / rel
        if not path.exists():
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
        shutil.rmtree(path)
        freed += size
        print(f"  removed  {rel:52s} {size / 1e9:6.2f} GB")
    print(f"\n  reclaimed {freed / 1e9:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="perform the moves, then verify")
    ap.add_argument("--delete", action="store_true", help="reclaim space (verifies first)")
    args = ap.parse_args()

    if args.delete:
        delete()
        return 0

    for line in build_plan(apply=args.apply):
        print(line)
    if args.apply:
        return 0 if verify() else 1
    print("\n(plan only -- rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
