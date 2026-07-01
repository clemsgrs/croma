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


def _toy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2"],
            "slide_id": ["sl0", "sl1", "sl2", "sl3"],
        }
    )


def _features() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def _install_noop_plots(monkeypatch) -> None:
    def fake_plot(*, out_path: Path, **kwargs: object) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"plot")

    for name in (
        "plot_bio_vs_confounder_scatter",
        "plot_croma_ltm_bars",
        "plot_croma_ltm_scatter",
        "plot_croma_m_sweep",
        "plot_croma_sample_distributions",
        "plot_croma_vs_mari_scatter",
        "plot_q_alpha_vs_croma_scatter",
        "plot_knn_bio_k_sweep",
        "plot_knn_confounder_k_sweep",
        "plot_mari_k_sweep",
        "plot_mari_vs_ri_scatter",
        "plot_ri_k_sweep",
        "plot_ri_mari_support",
    ):
        monkeypatch.setattr(bm, name, fake_plot)


def _write_precomputed_cache(
    *,
    manifest_path: Path,
    output_dir: Path,
    model: str,
    values: np.ndarray,
) -> None:
    manifest = bm.load_manifest(str(manifest_path), confounder_column="scanner_vendor")
    eval_manifest = bm._prepare_eval_manifest(
        manifest_df=manifest,
        dataset_name=str(manifest_path.stem),
        evaluation_design="dataset_wide",
    )
    embedding_manifest, _ = bm.build_embedding_source_manifest(eval_manifest)
    dataset_dir = output_dir / manifest_path.stem
    embeddings_dir = dataset_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_manifest_path = dataset_dir / "embedding_source_manifest.csv"
    embedding_manifest.to_csv(embedding_manifest_path, index=False)
    output_path = bm.ee._output_path_in_dir(manifest_path, embeddings_dir, model)
    np.save(output_path, values)
    output_path.with_suffix(".npy.json").write_text(
        json.dumps(
            {
                "manifest": str(embedding_manifest_path),
                "manifest_fingerprint": bm.manifest_fingerprint(embedding_manifest),
                "model_id": model,
                "extract": "precomputed",
                "mixed_precision": False,
                "n_samples": int(values.shape[0]),
                "embedding_dim": int(values.shape[1]),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_benchmark_accepts_unknown_models_when_precomputed_cache_exists(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    output_dir = tmp_path / "out"
    _toy_manifest().to_csv(manifest_path, index=False)

    _write_precomputed_cache(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model="PRISM",
        values=_features(),
    )
    _write_precomputed_cache(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model="TITAN",
        values=_features(),
    )

    def fake_registry() -> dict:
        return {"UNI": object()}

    def fail_embed_manifest(*args, **kwargs):
        raise AssertionError("benchmark should not try to extract cached unknown models")

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fail_embed_manifest)
    _install_noop_plots(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            "PRISM,TITAN",
            "--output-dir",
            str(output_dir),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "dataset_wide",
            "--k-max",
            "3",
            "--progress",
            "off",
        ],
    )

    assert bm.main() == 0

    metrics_df = pd.read_csv(output_dir / manifest_path.stem / "results" / "metrics.csv")
    assert metrics_df["model"].tolist() == ["PRISM", "TITAN"]
