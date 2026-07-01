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
import croma.metrics.neighbors as nb
from croma import MaRI, RI
from croma.metrics.pairs import normalize_manifest, resolve_manifest_subsets


def _paired_manifest() -> pd.DataFrame:
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


def _paired_features() -> np.ndarray:
    return np.asarray(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
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
        "plot_ri_mari_support",
        "plot_ri_k_sweep",
    ):
        monkeypatch.setattr(bm, name, fake_plot)


def test_paired_cached_artifacts_match_uncached_metric_outputs() -> None:
    manifest = _paired_manifest()
    features = _paired_features()
    k_values = [1, 3]
    selected_k = 1
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    canonical_manifest = normalize_manifest(
        manifest, confounder_column="scanner_vendor", source="manifest"
    )
    subsets = resolve_manifest_subsets(canonical_manifest)

    for metric_cls, extra_kwargs in ((RI, {}), (MaRI, {"tau": 0.2})):
        uncached = metric_cls._compute_artifacts(
            features=features,
            manifest=manifest,
            confounder_column="scanner_vendor",
            k_values=k_values,
            evaluation_design="paired_2x2",
            selected_k=selected_k,
            include_selected_result=True,
            warn_selected_result=False,
            **extra_kwargs,
        )
        cached_subsets = metric_cls._prepare_paired_subset_neighbor_cache(
            features=normalized,
            subsets=subsets,
            k_values=k_values,
        )
        cached = metric_cls._compute_artifacts_from_prepared_subsets(
            prepared_subsets=cached_subsets,
            dataset_name="toy",
            k_values=k_values,
            evaluation_design="paired_2x2",
            selected_k=selected_k,
            include_selected_result=True,
            warn_selected_result=False,
            **extra_kwargs,
        )

        assert cached.curve == uncached.curve
        assert uncached.result is not None
        assert cached.result is not None
        assert cached.result.k == uncached.result.k
        assert cached.result.value == uncached.result.value
        assert cached.result.std == uncached.result.std
        np.testing.assert_array_equal(
            cached.result.pair_values, uncached.result.pair_values
        )
        np.testing.assert_allclose(
            cached.result.sample_values,
            uncached.result.sample_values,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            cached.result.sample_values_aligned,
            uncached.result.sample_values_aligned,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            cached.result.occurrence_defined_mask,
            uncached.result.occurrence_defined_mask,
        )
        np.testing.assert_array_equal(
            cached.result.sample_undefined_types, uncached.result.sample_undefined_types
        )
        np.testing.assert_array_equal(
            cached.result.occurrence_subsets, uncached.result.occurrence_subsets
        )
        np.testing.assert_array_equal(
            cached.result.occurrence_source_indices,
            uncached.result.occurrence_source_indices,
        )


def test_benchmark_paired_prepares_neighbors_once_per_subset(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _paired_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"

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
        arr = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.85, 0.15, 0.0, 0.0],
                [0.15, 0.85, 0.0, 0.0],
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

    neighbor_prepare_calls = {"count": 0}
    original_prepare = nb._prepare_neighbors_with_meta

    def wrapped_prepare(*args, **kwargs):
        neighbor_prepare_calls["count"] += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    monkeypatch.setattr(nb, "_prepare_neighbors_with_meta", wrapped_prepare)
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
    assert neighbor_prepare_calls["count"] == 2

    per_sample_df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "per_sample_metrics.csv"
    )
    metrics_df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "metrics.csv"
    )
    assert len(per_sample_df) == 8
    assert per_sample_df["occurrence_index"].tolist() == list(range(8))
    assert set(per_sample_df["subset"]) == {"pair1", "pair2"}
    assert set(per_sample_df["source_sample_index"]) == set(range(8))
    assert set(metrics_df["evaluation_design"]) == {"paired_2x2"}


def _dataset_wide_manifest() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "sample_id": f"s{i}",
                "image_path": f"/tmp/s{i}.png",
                "label": "A" if i % 2 == 0 else "B",
                "scanner_vendor": "V1" if i < 4 else "V2",
                "slide_id": f"sl{i}",
                "dataset": "toy",
            }
        )
    return pd.DataFrame(rows)


def test_benchmark_dataset_wide_shares_one_neighbor_cache_across_ri_mari_tau(
    monkeypatch, tmp_path: Path
) -> None:
    """RI, MaRI and the tau-scale check must reuse a single dataset-wide neighbour cache.

    Per model on a cold cache the only neighbour preparations are the two (unpruned) kNN
    curves plus one shared RI/MaRI/tau cache -> 3 calls. Before the shared cache RI, MaRI
    and the tau check each prepared their own (5 calls); this pins the reuse.
    """
    manifest_path = tmp_path / "toy.csv"
    _dataset_wide_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"

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
        base = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
        conf_offset = {"V1": 0.0, "V2": 0.25}
        conf_col = "confounder" if "confounder" in manifest_df.columns else "scanner_vendor"
        arr = np.asarray(
            [
                base[row["label"]] + [conf_offset[row[conf_col]], 0.0]
                for _, row in manifest_df.iterrows()
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

    neighbor_prepare_calls = {"count": 0}
    original_prepare = nb._prepare_neighbors_with_meta

    def wrapped_prepare(*args, **kwargs):
        neighbor_prepare_calls["count"] += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    monkeypatch.setattr(nb, "_prepare_neighbors_with_meta", wrapped_prepare)
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
            "dataset_wide",
            "--k-max",
            "3",
            "--progress",
            "off",
        ],
    )

    assert bm.main() == 0
    # 2 unpruned kNN curves (bio + confounder) + 1 shared RI/MaRI/tau cache.
    assert neighbor_prepare_calls["count"] == 3

    metrics_df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "metrics.csv"
    )
    assert set(metrics_df["evaluation_design"]) == {"dataset_wide"}
