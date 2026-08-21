import hashlib
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts" / "studies", ROOT / "scripts" / "bench"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import pooling_sensitivity as study
import views as benchmark_views


def test_waiv_study_plan_freezes_models_benchmarks_and_operating_points() -> None:
    assert study.WAIV_STUDY_MODELS == {
        "Mascaret": study.StudyModelPlan(
            alternative="cls-mean-patch",
            alternative_width=3072,
            batch_size=32,
        ),
        "Phaet": study.StudyModelPlan(
            alternative="cls-mean-patch",
            alternative_width=2048,
            batch_size=64,
        ),
    }
    assert study.PATHOROB_STUDY_BENCHMARKS == {
        "pathorob-camelyon": study.StudyBenchmarkPlan(
            tileset="pathorob-camelyon",
            evaluation_design="all",
            fixed_k=11,
            biological_k_max=600,
            diagnostic_k_max=300,
        ),
        "pathorob-tcga-2x2": study.StudyBenchmarkPlan(
            tileset="pathorob-tcga-2x2",
            evaluation_design="paired_2x2",
            fixed_k=61,
            biological_k_max=1200,
            diagnostic_k_max=None,
        ),
        "pathorob-tcga-4x4": study.StudyBenchmarkPlan(
            tileset="pathorob-tcga-4x4",
            evaluation_design="all",
            fixed_k=71,
            biological_k_max=600,
            diagnostic_k_max=None,
        ),
        "pathorob-tolkach-esca": study.StudyBenchmarkPlan(
            tileset="pathorob-tolkach-esca",
            evaluation_design="all",
            fixed_k=61,
            biological_k_max=1000,
            diagnostic_k_max=None,
        ),
    }


def _write_canonical_panel(root: Path) -> dict[str, tuple[int, int]]:
    before: dict[str, tuple[int, int]] = {}
    for tileset in study.PATHOROB_TILESETS:
        directory = root / tileset
        directory.mkdir(parents=True, exist_ok=True)
        for model in study.STUDY_MODELS:
            matrix = directory / f"{model}.npy"
            sidecar = matrix.with_suffix(".npy.json")
            matrix.write_bytes(f"matrix:{tileset}:{model}".encode())
            sidecar.write_text(
                json.dumps({"tileset": tileset, "model": model}) + "\n",
                encoding="utf-8",
            )
            before[str(matrix)] = (matrix.stat().st_size, matrix.stat().st_mtime_ns)
            before[str(sidecar)] = (sidecar.stat().st_size, sidecar.stat().st_mtime_ns)
    return before


def test_preservation_baseline_records_exactly_the_twenty_study_pairs(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "output" / "embeddings"
    study_root = tmp_path / "output" / "studies" / "pooling-sensitivity"
    before = _write_canonical_panel(canonical_root)

    baseline_path = study.capture_preservation_baseline(
        canonical_root=canonical_root,
        study_root=study_root,
    )

    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["artifacts"]) == 40
    assert sum(item["kind"] == "matrix" for item in payload["artifacts"]) == 20
    assert sum(item["kind"] == "sidecar" for item in payload["artifacts"]) == 20
    for item in payload["artifacts"]:
        path = canonical_root / item["relative_path"]
        assert item == {
            "kind": "sidecar" if path.name.endswith(".npy.json") else "matrix",
            "relative_path": path.relative_to(canonical_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    assert {
        path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in before
    } == before


def test_per_occurrence_npz_is_byte_deterministic_with_fixed_zip_metadata() -> None:
    arrays = {
        "canonical_croma": np.array([0.25, -0.5], dtype=np.float64),
        "alternative_croma": np.array([0.5, -0.25], dtype=np.float64),
        "occurrence_index": np.array([0, 1], dtype=np.int64),
        "source_sample_index": np.array([3, 7], dtype=np.int64),
        "subset": np.array(["dataset", "dataset"]),
        "sample_id": np.array(["tile-a", "tile-b"]),
        "group_id": np.array(["slide-a", "slide-b"]),
    }

    first = study.deterministic_npz_bytes(arrays)
    second = study.deterministic_npz_bytes(dict(reversed(list(arrays.items()))))

    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == [f"{name}.npy" for name in sorted(arrays)]
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)


def test_compatible_study_bundle_rerun_performs_zero_target_writes(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "output" / "studies" / "pooling-sensitivity"
    files = {
        Path("results/comparisons.csv"): b"model,representation\nMascaret,canonical\n",
        Path("report.md"): b"# Pooling sensitivity\n",
    }

    assert study.publish_study_bundle(study_root, files) == "written"
    before = {relative: (study_root / relative).stat().st_mtime_ns for relative in files}

    assert study.publish_study_bundle(study_root, files) == "reused"
    assert {relative: (study_root / relative).stat().st_mtime_ns for relative in files} == before


def test_study_bundle_check_compares_bytes_without_touching_targets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study"
    files = {Path("report.md"): b"stable\n"}
    study.publish_study_bundle(root, files)
    target = root / "report.md"
    before = (target.read_bytes(), target.stat().st_mtime_ns)

    assert study.publish_study_bundle(root, files, check=True) == "checked"
    with pytest.raises(RuntimeError, match="check failed"):
        study.publish_study_bundle(root, {Path("report.md"): b"different\n"}, check=True)

    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_force_cannot_escape_the_study_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"

    with pytest.raises(ValueError, match="escapes"):
        study.publish_study_bundle(
            tmp_path / "study",
            {Path("../outside.txt"): b"forbidden"},
            force=True,
        )

    assert not outside.exists()


def test_occurrence_artifact_rejects_representation_identity_mismatch() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["tile-a", "tile-b"],
            "group_id": ["slide-a", "slide-b"],
        }
    )
    canonical = SimpleNamespace(
        sample_values_aligned=np.array([0.1, 0.2]),
        occurrence_source_indices=np.array([0, 1]),
        occurrence_subsets=np.array(["dataset", "dataset"]),
    )
    alternative = SimpleNamespace(
        sample_values_aligned=np.array([0.3, 0.4]),
        occurrence_source_indices=np.array([1, 0]),
        occurrence_subsets=np.array(["dataset", "dataset"]),
    )

    with pytest.raises(ValueError, match="occurrence identity mismatch"):
        study.build_occurrence_arrays(
            canonical=canonical,
            alternative=alternative,
            aligned_manifest=manifest,
        )


def _evaluation(representation: str, offset: float):
    return study.RepresentationEvaluation(
        representation=representation,
        fixed_k=11,
        biological_knn_bacc=0.70 + offset,
        confounder_knn_bacc=0.60 - offset,
        biological_kstar=21 + int(offset * 10),
        biological_kstar_bacc=0.75 + offset,
        diagnostic_kstar_300=11 + int(offset * 10),
        diagnostic_kstar_300_bacc=0.72 + offset,
        tau=0.2 + offset,
        ri=0.5 + offset,
        mari=0.55 + offset,
        support=0.8 + offset,
        ss_dominated_undefined_frac=0.1 - offset,
        oo_dominated_undefined_frac=0.05,
        mixed_undefined_frac=0.05,
        croma=0.1 + offset,
        croma_f0=0.4 - offset,
        croma_ltm10=-0.3 + offset,
        croma_result=None,
    )


def test_comparison_schema_has_one_shared_support_and_signed_absolute_deltas() -> None:
    canonical = _evaluation("canonical", 0.0)
    alternative = _evaluation("cls-mean-patch", 0.1)

    comparisons, rankings = study.build_comparison_frames(
        benchmark="pathorob-camelyon",
        tileset="pathorob-camelyon",
        model="Mascaret",
        canonical=canonical,
        alternative=alternative,
    )

    row = comparisons.iloc[0]
    assert row["canonical_support"] == pytest.approx(0.8)
    assert row["alternative_support"] == pytest.approx(0.9)
    assert row["delta_support"] == pytest.approx(0.1)
    assert row["abs_delta_support"] == pytest.approx(0.1)
    assert row["delta_croma"] == pytest.approx(0.1)
    assert row["abs_delta_croma"] == pytest.approx(0.1)
    assert not any("ri_support" in column or "mari_support" in column for column in row.index)
    assert not any("croma_support" in column for column in row.index)
    assert rankings["representation"].tolist() == ["cls-mean-patch", "canonical"]
    assert rankings["croma_rank"].tolist() == [1, 2]


def test_comparison_schema_leaves_camelyon_diagnostic_blank_for_other_benchmarks() -> None:
    canonical = replace(
        _evaluation("canonical", 0.0),
        diagnostic_kstar_300=None,
        diagnostic_kstar_300_bacc=None,
    )
    alternative = replace(
        _evaluation("cls-mean-patch", 0.1),
        diagnostic_kstar_300=None,
        diagnostic_kstar_300_bacc=None,
    )

    comparisons, rankings = study.build_comparison_frames(
        benchmark="pathorob-tcga-4x4",
        tileset="pathorob-tcga-4x4",
        model="Phaet",
        canonical=canonical,
        alternative=alternative,
    )

    assert (
        comparisons[
            [
                "canonical_diagnostic_kstar_300",
                "alternative_diagnostic_kstar_300",
                "delta_diagnostic_kstar_300",
                "abs_delta_diagnostic_kstar_300",
            ]
        ]
        .isna()
        .all(axis=None)
    )
    assert rankings["diagnostic_kstar_300"].isna().all()


def test_biological_kstars_use_production_sparse_grid_and_smallest_k_ties() -> None:
    scores = {k: 0.5 for k in [1, 3, 5, 7, 9, *range(11, 600, 10)]}
    scores[291] = 0.8
    scores[591] = 0.9

    production, diagnostic = study.select_biological_kstars(scores)

    assert production == (591, 0.9)
    assert diagnostic == (291, 0.8)

    tied = {k: 0.5 for k in scores}
    assert study.select_biological_kstars(tied) == ((1, 0.5), (1, 0.5))


def test_representation_evaluation_uses_fixed_k_auto_tau_and_total_croma() -> None:
    # Six rows per 2x2 cell make every k=11 neighbourhood contain exactly the
    # remaining biological class. The separated axes and fixed asymmetric offsets
    # avoid the cross-version distance ties produced by the former circle fixture.
    cells = [
        ("A", "V1", 4.0, 1.0),
        ("A", "V2", 4.0, -1.0),
        ("B", "V1", -4.0, 1.0),
        ("B", "V2", -4.0, -1.0),
    ]
    rows: list[list[float]] = []
    labels: list[str] = []
    confounders: list[str] = []
    for cell_index, (label, confounder, biological_axis, confounder_axis) in enumerate(cells):
        for occurrence in range(6):
            offset = occurrence + 1
            rows.append(
                [
                    biological_axis,
                    confounder_axis,
                    offset * 0.01,
                    offset**2 * 0.001 + cell_index * 0.0001,
                ]
            )
            labels.append(label)
            confounders.append(confounder)
    features = np.asarray(rows, dtype=np.float32)
    n = len(features)
    manifest = pd.DataFrame(
        {
            "sample_id": [f"tile-{i}" for i in range(n)],
            "image_path": [f"/tiles/{i}.png" for i in range(n)],
            "label": labels,
            "scanner_vendor": confounders,
            "group_id": [f"slide-{i}" for i in range(n)],
            "dataset": ["toy"] * n,
        }
    )

    result = study.evaluate_representation(
        representation="canonical",
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        fixed_k=11,
        production_k_max=20,
        diagnostic_k_max=10,
        headline_m=1,
        croma_start_k=11,
    )

    assert result.representation == "canonical"
    assert {
        key: value
        for key, value in result.__dict__.items()
        if key not in {"representation", "croma_result"}
    } == pytest.approx(
        {
            "fixed_k": 11,
            "biological_knn_bacc": 1.0,
            "confounder_knn_bacc": 0.0,
            "biological_kstar": 1,
            "biological_kstar_bacc": 1.0,
            "diagnostic_kstar_300": 1,
            "diagnostic_kstar_300_bacc": 1.0,
            "tau": 0.11765024065971375,
            "ri": 1.0,
            "mari": 1.0,
            "support": 1.0,
            "ss_dominated_undefined_frac": 0.0,
            "oo_dominated_undefined_frac": 0.0,
            "mixed_undefined_frac": 0.0,
            "croma": 0.8823441360626401,
            "croma_f0": 0.0,
            "croma_ltm10": 0.8823428586989946,
        }
    )
    assert result.croma_result.sample_values_aligned.shape == (n,)
    np.testing.assert_allclose(
        result.croma_result.sample_values_aligned[:5],
        [0.88234377, 0.88234286, 0.88234300, 0.88234449, 0.88234760],
    )


def test_paired_representation_evaluation_uses_occurrences_and_no_camelyon_diagnostic() -> None:
    cells = [
        ("A", "V1", 4.0, 1.0),
        ("A", "V2", 4.0, -1.0),
        ("B", "V1", -4.0, 1.0),
        ("B", "V2", -4.0, -1.0),
    ]
    rows: list[list[float]] = []
    manifest_rows: list[dict[str, str]] = []
    for subset in ("pair-1", "pair-2"):
        for cell_index, (label, confounder, biological_axis, confounder_axis) in enumerate(cells):
            for occurrence in range(6):
                offset = occurrence + 1
                rows.append(
                    [
                        biological_axis,
                        confounder_axis,
                        offset * 0.01,
                        offset**2 * 0.001 + cell_index * 0.0001,
                    ]
                )
                index = len(rows) - 1
                manifest_rows.append(
                    {
                        "sample_id": f"tile-{index}",
                        "image_path": f"/tiles/{index}.png",
                        "label": label,
                        "scanner_vendor": confounder,
                        "group_id": f"slide-{index}",
                        "dataset": "toy",
                        "subset": subset,
                    }
                )

    result = study.evaluate_representation(
        representation="canonical",
        features=np.asarray(rows, dtype=np.float32),
        manifest=pd.DataFrame(manifest_rows),
        confounder_column="scanner_vendor",
        evaluation_design="paired_2x2",
        fixed_k=11,
        production_k_max=20,
        diagnostic_k_max=None,
        headline_m=1,
        croma_start_k=11,
    )

    assert result.biological_kstar == 1
    assert result.diagnostic_kstar_300 is None
    assert result.diagnostic_kstar_300_bacc is None
    assert result.biological_knn_bacc == 1.0
    assert result.confounder_knn_bacc == 0.0
    assert result.ri == 1.0
    assert result.mari == 1.0
    assert result.support == 1.0
    assert result.croma == pytest.approx(0.8823441360626401)
    assert result.croma_result.evaluation_design == "paired_2x2"
    assert result.croma_result.evaluation_unit == "occurrence"
    assert result.croma_result.sample_values_aligned.shape == (48,)


def test_benchmark_view_accepts_explicit_embedding_and_manifest_roots(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "embeddings"
    tileset = canonical_root / "pathorob-camelyon"
    tileset.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "image_path": ["a.png", "b.png", "c.png"],
        }
    ).to_csv(tileset / "manifest.csv", index=False)
    np.save(tileset / "Mascaret.npy", np.arange(6).reshape(3, 2))
    eval_manifest = tmp_path / "eval.csv"
    pd.DataFrame(
        {
            "sample_id": ["c", "a"],
            "image_path": ["c.png", "a.png"],
            "label": ["tumor", "normal"],
            "medical_center": ["RUMC", "UMCU"],
            "group_id": ["slide-c", "slide-a"],
        }
    ).to_csv(eval_manifest, index=False)

    view = benchmark_views.load_view(
        "pathorob-camelyon",
        embeddings_root=canonical_root,
        eval_manifest_path=eval_manifest,
    )

    assert view.rows.tolist() == [2, 0]
    np.testing.assert_array_equal(view.features("Mascaret"), [[4, 5], [0, 1]])


def test_rendered_study_bundle_has_required_layout_and_reproducible_bootstrap() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["tile-a", "tile-b"],
            "group_id": ["slide-a", "slide-b"],
        }
    )
    identity = {
        "occurrence_source_indices": np.array([0, 1]),
        "occurrence_subsets": np.array(["dataset", "dataset"]),
    }
    canonical = _evaluation("canonical", 0.0)
    alternative = _evaluation("cls-mean-patch", 0.1)
    canonical = replace(
        canonical,
        croma_result=SimpleNamespace(sample_values_aligned=np.array([0.0, 0.2]), **identity),
    )
    alternative = replace(
        alternative,
        croma=0.3,
        croma_result=SimpleNamespace(sample_values_aligned=np.array([0.2, 0.4]), **identity),
    )

    first = study.render_study_bundle(
        benchmark="pathorob-camelyon",
        tileset="pathorob-camelyon",
        model="Mascaret",
        canonical=canonical,
        alternative=alternative,
        aligned_manifest=manifest,
        provenance_inputs={"canonical_sha256": "a" * 64},
        replay_commands=["python scripts/studies/pooling_sensitivity.py"],
        n_boot=4,
    )
    second = study.render_study_bundle(
        benchmark="pathorob-camelyon",
        tileset="pathorob-camelyon",
        model="Mascaret",
        canonical=canonical,
        alternative=alternative,
        aligned_manifest=manifest,
        provenance_inputs={"canonical_sha256": "a" * 64},
        replay_commands=["python scripts/studies/pooling_sensitivity.py"],
        n_boot=4,
    )

    assert first == second
    assert set(first) == {
        Path("results/comparisons.csv"),
        Path("results/rankings.csv"),
        Path("per-occurrence/pathorob-camelyon/Mascaret.npz"),
        Path("run-provenance.json"),
        Path("report.md"),
    }
    provenance = json.loads(first[Path("run-provenance.json")])
    assert provenance["croma_version"] == "1.0.0"
    assert provenance["bootstrap"] == {
        "grouping": "shared-group_id",
        "level": 0.95,
        "method": "numpy-linear-percentile",
        "n_boot": 4,
        "seed": 0,
        "contrast": "alternative-minus-canonical-headline-croma",
    }
    comparisons = pd.read_csv(io.BytesIO(first[Path("results/comparisons.csv")]))
    assert comparisons.loc[0, "croma_delta_ci_point"] == pytest.approx(0.2)
    assert comparisons.loc[0, "median_paired_occurrence_croma_delta"] == pytest.approx(0.2)
    assert b"fixed k=11" in first[Path("report.md")]


def test_paired_bootstrap_recomputes_median_of_subset_medians_from_one_shared_draw() -> None:
    canonical = np.array([0.0, 2.0, 100.0, 102.0, 104.0, 106.0])
    alternative = np.array([10.0, 12.0, 90.0, 92.0, 94.0, 96.0])
    groups = np.array(["g1", "g2", "g3", "g4", "g5", "g6"])
    subsets = np.array(["A", "A", "B", "B", "B", "B"])

    result = study.paired_cluster_bootstrap_delta(
        canonical,
        alternative,
        groups,
        subset_ids=subsets,
        n_boot=4,
        seed=0,
    )

    assert result.point == 0.0
    assert result.lo == -9.25
    assert result.hi == 0.0
    assert np.median(alternative) - np.median(canonical) == -10.0


def test_paired_bundle_uses_subset_balanced_headline_and_labels_pooled_medians_descriptive() -> (
    None
):
    subsets = np.array(["A", "A", "B", "B", "B", "B"])
    sources = np.arange(6, dtype=np.int64)
    manifest = pd.DataFrame(
        {
            "sample_id": ["repeat", "repeat", "b1", "b2", "b3", "b4"],
            "group_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "source_sample_index": sources,
            "subset": subsets,
        }
    )
    identity = {
        "occurrence_source_indices": sources,
        "occurrence_subsets": subsets,
    }
    canonical = replace(
        _evaluation("canonical", 0.0),
        croma=52.0,
        croma_result=SimpleNamespace(
            sample_values_aligned=np.array([0.0, 2.0, 100.0, 102.0, 104.0, 106.0]),
            **identity,
        ),
    )
    alternative = replace(
        _evaluation("cls-mean-patch", 0.1),
        croma=52.0,
        croma_result=SimpleNamespace(
            sample_values_aligned=np.array([10.0, 12.0, 90.0, 92.0, 94.0, 96.0]),
            **identity,
        ),
    )

    files = study.render_study_bundle(
        benchmark="pathorob-tcga-2x2",
        tileset="pathorob-tcga-2x2",
        model="Phaet",
        canonical=canonical,
        alternative=alternative,
        aligned_manifest=manifest,
        provenance_inputs={},
        replay_commands=[],
        n_boot=4,
    )

    comparisons = pd.read_csv(io.BytesIO(files[Path("results/comparisons.csv")]))
    row = comparisons.iloc[0]
    assert row["croma_delta_ci_point"] == 0.0
    assert row["canonical_pooled_occurrence_croma"] == 101.0
    assert row["alternative_pooled_occurrence_croma"] == 91.0
    assert row["delta_pooled_occurrence_croma"] == -10.0
    assert row["median_paired_occurrence_croma_delta"] == -10.0


def test_panel_bundle_is_order_independent_and_aggregates_each_run_once() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["tile-a", "tile-b"],
            "group_id": ["slide-a", "slide-b"],
            "source_sample_index": [0, 1],
            "subset": ["dataset", "dataset"],
        }
    )
    identity = {
        "occurrence_source_indices": np.array([0, 1]),
        "occurrence_subsets": np.array(["dataset", "dataset"]),
    }

    def run(benchmark: str, model: str, offset: float) -> study.StudyRun:
        canonical = replace(
            _evaluation("canonical", 0.0),
            croma_result=SimpleNamespace(sample_values_aligned=np.array([0.0, 0.2]), **identity),
        )
        alternative = replace(
            _evaluation("cls-mean-patch", offset),
            croma=0.1 + offset,
            croma_result=SimpleNamespace(
                sample_values_aligned=np.array([offset, 0.2 + offset]), **identity
            ),
        )
        return study.StudyRun(
            benchmark=benchmark,
            tileset=benchmark,
            model=model,
            canonical=canonical,
            alternative=alternative,
            aligned_manifest=manifest,
            provenance_inputs={"matrix": f"{benchmark}-{model}"},
        )

    runs = [
        run("pathorob-tolkach-esca", "Phaet", 0.2),
        run("pathorob-camelyon", "Mascaret", 0.1),
    ]
    first = study.render_panel_bundle(runs=runs, replay_commands=["replay"], n_boot=4)
    second = study.render_panel_bundle(
        runs=list(reversed(runs)), replay_commands=["replay"], n_boot=4
    )

    assert first == second
    assert {path for path in first if path.parts[0] == "per-occurrence"} == {
        Path("per-occurrence/pathorob-camelyon/Mascaret.npz"),
        Path("per-occurrence/pathorob-tolkach-esca/Phaet.npz"),
    }
    comparisons = pd.read_csv(io.BytesIO(first[Path("results/comparisons.csv")]))
    assert comparisons[["benchmark", "model"]].to_dict("records") == [
        {"benchmark": "pathorob-camelyon", "model": "Mascaret"},
        {"benchmark": "pathorob-tolkach-esca", "model": "Phaet"},
    ]
    provenance = json.loads(first[Path("run-provenance.json")])
    assert [(item["benchmark"], item["model"]) for item in provenance["runs"]] == [
        ("pathorob-camelyon", "Mascaret"),
        ("pathorob-tolkach-esca", "Phaet"),
    ]


def test_baseline_only_cli_supports_no_write_check(tmp_path: Path) -> None:
    canonical_root = tmp_path / "output" / "embeddings"
    study_root = tmp_path / "output" / "studies" / "pooling-sensitivity"
    _write_canonical_panel(canonical_root)

    args = [
        "--canonical-root",
        str(canonical_root),
        "--study-root",
        str(study_root),
        "--baseline-only",
    ]
    assert study.main(args) == 0
    baseline = study_root / "preservation-baseline.json"
    before = (baseline.read_bytes(), baseline.stat().st_mtime_ns)

    assert study.main([*args, "--check"]) == 0
    assert (baseline.read_bytes(), baseline.stat().st_mtime_ns) == before


def test_legacy_eval_manifest_cli_keeps_the_issue_150_tracer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(study, "capture_preservation_baseline", lambda **kwargs: None)
    monkeypatch.setattr(
        study,
        "run_mascaret_camelyon",
        lambda **kwargs: calls.append(kwargs) or {},
    )
    canonical_root = tmp_path / "embeddings"
    study_root = tmp_path / "study"
    eval_manifest = tmp_path / "camelyon.csv"

    assert (
        study.main(
            [
                "--canonical-root",
                str(canonical_root),
                "--study-root",
                str(study_root),
                "--eval-manifest",
                str(eval_manifest),
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--force",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "canonical_root": canonical_root,
            "study_root": study_root,
            "eval_manifest_path": eval_manifest,
            "device_arg": "cpu",
            "batch_size": 32,
            "num_workers": 0,
            "check": False,
            "force": True,
        }
    ]


def test_alternative_extraction_is_study_owned_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "output" / "embeddings"
    study_root = tmp_path / "output" / "studies" / "pooling-sensitivity"
    directory = canonical_root / "pathorob-camelyon"
    directory.mkdir(parents=True)
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["/tiles/a.png", "/tiles/b.png"],
            "label": ["normal", "tumor"],
            "confounder": ["RUMC", "UMCU"],
            "group_id": ["slide-a", "slide-b"],
        }
    )
    manifest.to_csv(directory / "manifest.csv", index=False)
    canonical = directory / "Mascaret.npy"
    np.save(canonical, np.zeros((2, 1536), dtype=np.float32))
    canonical_before = (canonical.read_bytes(), canonical.stat().st_mtime_ns)
    writes: list[Path] = []

    def fake_embed_manifest(*, output_path, artifact_contract, **kwargs):
        writes.append(Path(output_path))
        study.extraction.publish_embedding_artifact(
            Path(output_path),
            np.zeros((2, 3072), dtype=np.float32),
            artifact_contract,
        )
        return Path(output_path), (2, 3072)

    monkeypatch.setattr(study.extraction, "embed_manifest", fake_embed_manifest)
    kwargs = dict(
        canonical_root=canonical_root,
        study_root=study_root,
        tileset="pathorob-camelyon",
        model="Mascaret",
        representation="cls-mean-patch",
        batch_size=2,
        num_workers=0,
        device_arg="cpu",
    )

    target, status = study.extract_study_representation(**kwargs)
    target_before = (target.read_bytes(), target.stat().st_mtime_ns)
    assert status == "written"
    assert target == study_root / "embeddings/pathorob-camelyon/Mascaret/cls-mean-patch.npy"
    assert [path.name for path in directory.glob("*.npy")] == ["Mascaret.npy"]

    assert study.extract_study_representation(**kwargs) == (target, "reused")
    assert study.extract_study_representation(**kwargs, force=True) == (target, "reused")
    assert study.extract_study_representation(**kwargs, check=True) == (
        target,
        "checked",
    )
    assert (target.read_bytes(), target.stat().st_mtime_ns) == target_before
    assert (canonical.read_bytes(), canonical.stat().st_mtime_ns) == canonical_before
    assert writes[0] == target
    assert writes[1] != target


def test_alternative_extraction_rejects_a_compatible_nonfinite_matrix(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "output" / "embeddings"
    study_root = tmp_path / "output" / "studies" / "pooling-sensitivity"
    directory = canonical_root / "pathorob-camelyon"
    directory.mkdir(parents=True)
    manifest_path = directory / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "image_path": ["/tiles/a.png"],
            "label": ["normal"],
            "confounder": ["RUMC"],
            "group_id": ["slide-a"],
        }
    ).to_csv(manifest_path, index=False)
    np.save(directory / "Mascaret.npy", np.zeros((1, 1536), dtype=np.float32))
    target = study.study_embedding_path(
        study_root=study_root,
        tileset="pathorob-camelyon",
        model="Mascaret",
        representation="cls-mean-patch",
    )
    contract = study.extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=study._build_model_registry()["Mascaret"],
        batch_size=1,
        device_arg="cpu",
        pooling="cls-mean-patch",
    )
    study.extraction.publish_embedding_artifact(
        target,
        np.full((1, 3072), np.nan, dtype=np.float32),
        contract,
    )

    with pytest.raises(RuntimeError, match="finite FP32"):
        study.extract_study_representation(
            canonical_root=canonical_root,
            study_root=study_root,
            tileset="pathorob-camelyon",
            model="Mascaret",
            representation="cls-mean-patch",
            batch_size=1,
            num_workers=0,
            device_arg="cpu",
        )


def test_waiv_panel_extraction_inventory_is_exact_and_uses_model_batch_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_extract(**kwargs):
        calls.append(kwargs)
        target = study.study_embedding_path(
            study_root=kwargs["study_root"],
            tileset=kwargs["tileset"],
            model=kwargs["model"],
            representation=kwargs["representation"],
        )
        return target, "reused" if kwargs["model"] == "Mascaret" else "written"

    monkeypatch.setattr(study, "extract_study_representation", fake_extract)
    study_root = tmp_path / "studies" / "pooling-sensitivity"

    inventory = study.extract_waiv_panel(
        canonical_root=tmp_path / "embeddings",
        study_root=study_root,
        device_arg="cuda",
        num_workers=3,
        check=False,
        force=False,
    )

    assert len(inventory) == 8
    assert set(inventory) == {
        (benchmark, model)
        for benchmark in study.PATHOROB_STUDY_BENCHMARKS
        for model in study.WAIV_STUDY_MODELS
    }
    assert {call["batch_size"] for call in calls if call["model"] == "Mascaret"} == {32}
    assert {call["batch_size"] for call in calls if call["model"] == "Phaet"} == {64}
    assert all(call["representation"] == "cls-mean-patch" for call in calls)
    assert inventory[("pathorob-camelyon", "Mascaret")] == (
        study_root / "embeddings/pathorob-camelyon/Mascaret/cls-mean-patch.npy",
        "reused",
    )


@pytest.mark.parametrize("member", ["matrix", "sidecar"])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_alternative_extraction_refuses_canonical_aliases_without_writes(
    tmp_path: Path, member: str, link_kind: str
) -> None:
    canonical_root = tmp_path / "output" / "embeddings"
    study_root = tmp_path / "output" / "studies" / "pooling-sensitivity"
    directory = canonical_root / "pathorob-camelyon"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "image_path": ["/tiles/a.png"],
            "label": ["normal"],
            "confounder": ["RUMC"],
            "group_id": ["slide-a"],
        }
    ).to_csv(directory / "manifest.csv", index=False)
    canonical = directory / "Mascaret.npy"
    np.save(canonical, np.zeros((1, 1536), dtype=np.float32))
    canonical_sidecar = canonical.with_suffix(".npy.json")
    canonical_sidecar.write_text('{"canonical": true}\n', encoding="utf-8")
    protected = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (canonical, canonical_sidecar)
    }
    target = study.study_embedding_path(
        study_root=study_root,
        tileset="pathorob-camelyon",
        model="Mascaret",
        representation="cls-mean-patch",
    )
    target.parent.mkdir(parents=True)
    source = canonical if member == "matrix" else canonical_sidecar
    alias = target if member == "matrix" else target.with_suffix(".npy.json")
    if link_kind == "symlink":
        alias.symlink_to(source)
    else:
        alias.hardlink_to(source)

    with pytest.raises(RuntimeError, match="symlink|hard-link|aliases"):
        study.extract_study_representation(
            canonical_root=canonical_root,
            study_root=study_root,
            tileset="pathorob-camelyon",
            model="Mascaret",
            representation="cls-mean-patch",
            batch_size=1,
            num_workers=0,
            device_arg="cpu",
        )

    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected} == protected


def test_study_mapping_assigns_each_model_family_its_only_alternative() -> None:
    assert study.STUDY_ALTERNATIVE_POOLING == {
        "Mascaret": "cls-mean-patch",
        "Phaet": "cls-mean-patch",
        "RudolfV 2": "cls-only",
        "RudolfV 2-B": "cls-only",
        "RudolfV 2-S": "cls-only",
    }


def test_canonical_loader_rejects_mismatched_sidecar_contract(tmp_path: Path) -> None:
    directory = tmp_path / "embeddings" / "pathorob-camelyon"
    directory.mkdir(parents=True)
    manifest_path = directory / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "image_path": ["a.png"],
            "label": ["normal"],
            "confounder": ["RUMC"],
            "group_id": ["slide-a"],
        }
    ).to_csv(manifest_path, index=False)
    canonical_path = directory / "Mascaret.npy"
    contract = study.extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=study._build_model_registry()["Mascaret"],
        batch_size=1,
        device_arg="cpu",
    )
    study.extraction.publish_embedding_artifact(
        canonical_path,
        np.zeros((1, 1536), dtype=np.float32),
        contract,
    )
    sidecar = canonical_path.with_suffix(".npy.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["checkpoint_revision"] = "0" * 40
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(study.ArtifactCompatibilityError, match="checkpoint_revision"):
        study._load_validated_canonical_matrix(
            canonical_path=canonical_path,
            manifest_path=manifest_path,
            batch_size=1,
            device_arg="cpu",
        )
