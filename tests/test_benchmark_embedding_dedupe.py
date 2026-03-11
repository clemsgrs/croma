import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm


def _repeated_subset_manifest() -> pd.DataFrame:
    base_rows = [
        ("s0", "/tmp/s0.png", "A", "V1", "sl0"),
        ("s1", "/tmp/s1.png", "A", "V2", "sl1"),
        ("s2", "/tmp/s2.png", "B", "V1", "sl2"),
        ("s3", "/tmp/s3.png", "B", "V2", "sl3"),
    ]
    rows: list[dict[str, str]] = []
    for subset in ("pair1", "pair2"):
        for sample_id, image_path, label, confounder, slide_id in base_rows:
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "label": label,
                    "scanner_vendor": confounder,
                    "slide_id": slide_id,
                    "subset": subset,
                    "dataset": "toy",
                }
            )
    return pd.DataFrame(rows)


def _install_noop_plots(monkeypatch) -> None:
    def fake_plot(*, out_path: Path, **kwargs: object) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"plot")

    for name in (
        "plot_bio_vs_confounder_scatter",
        "plot_ccmr_ltm_comparison",
        "plot_ccmr_m_sweep_with_ltm",
        "plot_ccmr_sample_distributions",
        "plot_ccmr_vs_mari_scatter",
        "plot_knn_bio_k_sweep",
        "plot_knn_confounder_k_sweep",
        "plot_mari_k_sweep",
        "plot_mari_vs_ri_scatter",
        "plot_ri_k_sweep",
    ):
        monkeypatch.setattr(bm, name, fake_plot)


def test_benchmark_embeds_unique_source_samples_once(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _repeated_subset_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    embed_calls: list[dict[str, object]] = []

    def fake_registry() -> dict:
        return {"M1": object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        manifest_df = pd.read_csv(manifest_path, dtype=str)
        embed_calls.append(
            {
                "manifest_path": str(manifest_path),
                "rows": len(manifest_df),
                "sample_ids": manifest_df["sample_id"].tolist(),
            }
        )
        arr = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        output_path.with_suffix(".npy.json").write_text(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "manifest_fingerprint": bm.manifest_fingerprint(manifest_df),
                    "model_id": "fake",
                    "extract": "cls",
                    "mixed_precision": False,
                    "n_samples": int(arr.shape[0]),
                    "embedding_dim": int(arr.shape[1]),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    _install_noop_plots(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            "M1",
            "--output-dir",
            str(output_dir),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "paired_2x2",
            "--k-max",
            "3",
            "--progress",
            "off",
        ],
    )

    assert bm.main() == 0
    assert len(embed_calls) == 1
    assert embed_calls[0]["rows"] == 4
    assert embed_calls[0]["sample_ids"] == ["s0", "s1", "s2", "s3"]

    per_sample_df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "per_sample_metrics.csv"
    )
    assert len(per_sample_df) == 8
    assert set(per_sample_df["subset"]) == {"pair1", "pair2"}
    assert int((per_sample_df["sample_id"] == "s0").sum()) == 2
