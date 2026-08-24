import hashlib
import io
import json
import shlex
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context, get_config

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


def test_rudolfv2_study_plan_is_the_frozen_three_by_four_matrix() -> None:
    model_ids = ("RudolfV 2", "RudolfV 2-B", "RudolfV 2-S")
    runs = study.build_study_plan(
        model_ids=model_ids,
        model_plans=study.RUDOLFV2_STUDY_MODELS,
    )

    assert len(runs) == 12
    assert [run.model_id for run in runs] == [model_id for model_id in model_ids for _ in range(4)]
    assert [
        (
            run.benchmark_name,
            run.benchmark.fixed_k,
            run.benchmark.biological_k_max,
            run.benchmark.diagnostic_k_max,
        )
        for run in runs[:4]
    ] == [
        ("pathorob-camelyon", 11, 600, 300),
        ("pathorob-tcga-2x2", 61, 1200, None),
        ("pathorob-tcga-4x4", 71, 600, None),
        ("pathorob-tolkach-esca", 61, 1000, None),
    ]
    assert [
        (
            run.model_id,
            run.identity.published_name,
            run.identity.variant_role,
            run.identity.parent_registry_id,
            run.identity.canonical_width,
            run.model_plan.alternative_width,
            run.model_plan.batch_size,
            run.model_plan.alternative,
        )
        for run in runs[::4]
    ] == [
        ("RudolfV 2", "RudolfV-2", "teacher", None, 3072, 1536, 32, "cls-only"),
        (
            "RudolfV 2-B",
            "RudolfV-2-B",
            "distilled-student",
            "RudolfV 2",
            1536,
            768,
            32,
            "cls-only",
        ),
        (
            "RudolfV 2-S",
            "RudolfV-2-S",
            "distilled-student",
            "RudolfV 2",
            768,
            384,
            64,
            "cls-only",
        ),
    ]


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


def test_preservation_baseline_reads_an_exact_access_mirror_but_keeps_logical_paths(
    tmp_path: Path,
) -> None:
    logical_root = tmp_path / "unavailable-logical-embeddings"
    access_root = tmp_path / "local-access-embeddings"
    study_root = tmp_path / "logical-study"
    before = _write_canonical_panel(access_root)
    direct_baseline = study.capture_preservation_baseline(
        canonical_root=access_root,
        study_root=tmp_path / "direct-study",
    )

    baseline = study.capture_preservation_baseline(
        canonical_root=logical_root,
        canonical_access_root=access_root,
        study_root=study_root,
    )

    assert baseline == study_root / study.PRESERVATION_BASELINE_NAME
    assert baseline.read_bytes() == direct_baseline.read_bytes()
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    for item in payload["artifacts"]:
        access_path = access_root / item["relative_path"]
        assert item["sha256"] == hashlib.sha256(access_path.read_bytes()).hexdigest()
        assert item["mtime_ns"] == access_path.stat().st_mtime_ns
    assert not logical_root.exists()
    assert study.verify_preservation_baseline(
        canonical_root=logical_root,
        canonical_access_root=access_root,
        study_root=study_root,
    ) == {"artifacts": 40, "matrices": 20, "sidecars": 20}
    assert {
        path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in before
    } == before


def test_preservation_baseline_rejects_an_incomplete_access_mirror(tmp_path: Path) -> None:
    access_root = tmp_path / "local-access-embeddings"
    _write_canonical_panel(access_root)
    missing = access_root / "pathorob-camelyon/RudolfV 2-S.npy.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=str(missing)):
        study.capture_preservation_baseline(
            canonical_root=tmp_path / "logical-embeddings",
            canonical_access_root=access_root,
            study_root=tmp_path / "study",
        )


def test_file_provenance_hashes_access_bytes_but_reports_the_logical_path(
    tmp_path: Path,
) -> None:
    logical = tmp_path / "logical" / "matrix.npy"
    access = tmp_path / "access" / "matrix.npy"
    access.parent.mkdir()
    access.write_bytes(b"exact-local-mirror")

    assert study._file_provenance(logical, access_path=access) == {
        "path": str(logical.resolve()),
        "sha256": hashlib.sha256(b"exact-local-mirror").hexdigest(),
        "size": len(b"exact-local-mirror"),
        "mtime_ns": access.stat().st_mtime_ns,
    }
    assert not logical.exists()


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


def test_force_rewrites_only_incompatible_bundle_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "study"
    initial = {
        Path("per-occurrence/waiv.npz"): b"protected-waiv-bytes",
        Path("report.md"): b"old report\n",
    }
    study.publish_study_bundle(root, initial)
    protected = root / "per-occurrence/waiv.npz"
    protected_before = (protected.read_bytes(), protected.stat().st_mtime_ns)
    writes: list[Path] = []
    atomic_write = study._atomic_write

    def recording_write(path: Path, payload: bytes) -> None:
        writes.append(path)
        atomic_write(path, payload)

    monkeypatch.setattr(study, "_atomic_write", recording_write)

    assert (
        study.publish_study_bundle(
            root,
            {**initial, Path("report.md"): b"new report\n"},
            force=True,
        )
        == "forced"
    )
    assert writes == [root / "report.md"]
    assert (protected.read_bytes(), protected.stat().st_mtime_ns) == protected_before
    assert (root / "report.md").read_bytes() == b"new report\n"


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


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("occurrence_index", 9),
        ("source_sample_index", 9),
        ("subset", "other"),
        ("sample_id", "other-sample"),
        ("group_id", "other-group"),
    ],
)
def test_paired_occurrence_identity_rejects_each_exact_component(
    field: str,
    replacement: int | str,
) -> None:
    canonical = pd.DataFrame(
        {
            "occurrence_index": [0, 1],
            "source_sample_index": [4, 7],
            "subset": ["A", "B"],
            "sample_id": ["repeat", "repeat"],
            "group_id": ["slide-a", "slide-b"],
        }
    )
    alternative = canonical.copy()
    alternative.loc[1, field] = replacement

    with pytest.raises(ValueError, match=field):
        study.validate_occurrence_identity(
            canonical_identity=canonical,
            alternative_identity=alternative,
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


def test_rudolf_machine_outputs_keep_registry_teacher_student_identity() -> None:
    canonical = _evaluation("canonical", 0.0)
    alternative = _evaluation("cls-only", 0.1)

    comparisons, rankings = study.build_comparison_frames(
        benchmark="pathorob-camelyon",
        tileset="pathorob-camelyon",
        model="RudolfV 2-B",
        canonical=canonical,
        alternative=alternative,
    )

    expected = {
        "model": "RudolfV 2-B",
        "variant_role": "distilled-student",
        "parent_model": "RudolfV 2",
    }
    assert comparisons.loc[0, list(expected)].to_dict() == expected
    assert rankings[list(expected)].to_dict("records") == [expected, expected]
    assert "published_name" not in comparisons
    assert "published_name" not in rankings

    comparison = comparisons.iloc[0].copy()
    comparison["croma_delta_ci_point"] = 0.1
    comparison["croma_delta_ci_lo"] = 0.01
    comparison["croma_delta_ci_hi"] = 0.2
    comparison["croma_delta_supported"] = True
    comparison["median_paired_occurrence_croma_delta"] = 0.1
    report = study._render_report(
        benchmark="pathorob-camelyon",
        model="RudolfV 2-B",
        canonical=canonical,
        alternative=alternative,
        comparison=comparison,
    ).decode()
    assert "Model: `RudolfV-2-B`" in report
    assert "distilled student of `RudolfV-2`" in report


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


def test_paired_representation_evaluation_uses_occurrences_and_no_camelyon_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    streamed_cache_calls: list[tuple[int, int, bool]] = []
    original_iterator = study.RI._iter_paired_subset_neighbor_cache

    def record_iterator(**kwargs):
        streamed_cache_calls.append(
            (
                len(kwargs["subsets"]),
                max(kwargs["k_values"]),
                kwargs["assume_normalized"],
            )
        )
        yield from original_iterator(**kwargs)

    monkeypatch.setattr(study.RI, "_iter_paired_subset_neighbor_cache", record_iterator)

    result = study.evaluate_representation(
        representation="canonical",
        features=np.asarray(rows, dtype=np.float32),
        manifest=pd.DataFrame(manifest_rows),
        confounder_column="scanner_vendor",
        evaluation_design="paired_2x2",
        fixed_k=5,
        production_k_max=20,
        diagnostic_k_max=None,
        headline_m=1,
        croma_start_k=11,
    )

    assert result.biological_kstar == 1
    assert result.diagnostic_kstar_300 is None
    assert result.diagnostic_kstar_300_bacc is None
    assert result.biological_knn_bacc == 1.0
    assert result.confounder_knn_bacc == 1.0
    assert result.ri == 0.5
    assert result.mari == 0.5
    assert result.support == 0.0
    assert result.croma == pytest.approx(0.8823441360626401)
    assert result.croma_result.evaluation_design == "paired_2x2"
    assert result.croma_result.evaluation_unit == "occurrence"
    assert result.croma_result.sample_values_aligned.shape == (48,)
    assert streamed_cache_calls == [(2, 11, False)]


def test_streamed_paired_evaluation_retains_only_fixed_k_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_cache = study._PreparedNeighborSubset(
        subset_id="pair-1",
        source_indices=np.arange(4, dtype=np.int64),
        labels=np.asarray([0, 0, 1, 1]),
        centers=np.asarray([0, 1, 0, 1]),
        group_ids=np.asarray(["g0", "g1", "g2", "g3"]),
        neigh_idx=np.asarray(
            [
                [1, 2, 3],
                [0, 2, 3],
                [3, 0, 1],
                [2, 0, 1],
            ],
            dtype=np.int64,
        ),
        neigh_dist=np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
            ],
            dtype=float,
        ),
        valid_counts=np.full(4, 3, dtype=np.int64),
    )

    def fake_iterator(**kwargs):
        assert kwargs["k_values"] == [1, 3]
        assert kwargs["assume_normalized"] is False
        yield full_cache

    monkeypatch.setattr(study.RI, "_iter_paired_subset_neighbor_cache", fake_iterator)

    retained, biological_scores = study._prepare_streamed_paired_evaluation(
        features=np.zeros((4, 2), dtype=np.float32),
        subsets=[],
        production_k_values=[1, 3],
        fixed_k=1,
        warn_context="toy biological k*",
    )

    assert biological_scores == {1: 1.0, 3: 0.0}
    assert len(retained) == 1
    assert retained[0].neigh_idx.shape == (4, 1)
    assert retained[0].neigh_dist.shape == (4, 1)
    np.testing.assert_array_equal(retained[0].valid_counts, np.ones(4, dtype=np.int64))
    assert not np.shares_memory(retained[0].neigh_idx, full_cache.neigh_idx)
    assert not np.shares_memory(retained[0].neigh_dist, full_cache.neigh_dist)


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


def test_evaluate_only_cli_validates_existing_inventory_without_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "embeddings"
    study_root = tmp_path / "study"
    canonical_access_root = tmp_path / "access-embeddings"
    study_access_root = tmp_path / "access-study"
    eval_manifest_root = tmp_path / "logical-eval-manifests"
    eval_manifest_access_root = tmp_path / "access-eval-manifests"
    calls: dict[str, object] = {}
    monkeypatch.setattr(study, "capture_preservation_baseline", lambda **kwargs: None)
    monkeypatch.setattr(study, "verify_preservation_baseline", lambda **kwargs: None)
    monkeypatch.setattr(
        study,
        "extract_pooling_panel",
        lambda **kwargs: pytest.fail("evaluate-only must not extract representations"),
    )
    monkeypatch.setattr(
        study,
        "build_study_inventory",
        lambda **kwargs: calls.setdefault("validated", kwargs),
    )

    def fake_evaluate(**kwargs):
        calls["inventory"] = kwargs["inventory"]
        calls["evaluate"] = kwargs
        return ["run"]

    monkeypatch.setattr(study, "evaluate_pooling_panel_isolated", fake_evaluate)
    monkeypatch.setattr(
        study,
        "render_panel_bundle",
        lambda **kwargs: calls.setdefault("replay", kwargs["replay_commands"])
        and {Path("report.md"): b"report\n"},
    )
    monkeypatch.setattr(
        study,
        "publish_study_bundle",
        lambda *args, **kwargs: calls.setdefault("publish", kwargs) or "checked",
    )

    assert (
        study.main(
            [
                "--canonical-root",
                str(canonical_root),
                "--study-root",
                str(study_root),
                "--canonical-access-root",
                str(canonical_access_root),
                "--study-access-root",
                str(study_access_root),
                "--eval-manifest-root",
                str(eval_manifest_root),
                "--eval-manifest-access-root",
                str(eval_manifest_access_root),
                "--models",
                "RudolfV 2-S",
                "--benchmarks",
                "pathorob-camelyon",
                "--evaluate-only",
                "--check",
            ]
        )
        == 0
    )
    target = study_root / "embeddings/pathorob-camelyon/RudolfV 2-S/cls-only.npy"
    assert calls["inventory"] == {("pathorob-camelyon", "RudolfV 2-S"): (target, "reused")}
    assert calls["validated"]["canonical_access_root"] == canonical_access_root
    assert calls["validated"]["study_access_root"] == study_access_root
    assert calls["evaluate"]["canonical_access_root"] == canonical_access_root
    assert calls["evaluate"]["study_access_root"] == study_access_root
    assert calls["evaluate"]["eval_manifest_access_root"] == eval_manifest_access_root
    assert all("access-" not in command for command in calls["replay"])
    for command in calls["replay"]:
        arguments = shlex.split(command)
        assert arguments[arguments.index("--models") + 1 : arguments.index("--benchmarks")] == [
            "RudolfV 2-S"
        ]
    assert calls["publish"] == {"check": True, "force": False}


def test_isolated_cell_worker_bounds_neighbor_working_memory_during_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_working_memory: list[int] = []
    sentinel = object()

    def fake_evaluate(**kwargs):
        observed_working_memory.append(get_config()["working_memory"])
        return [sentinel]

    monkeypatch.setattr(study, "evaluate_pooling_panel", fake_evaluate)
    request = study.StudyCellRequest(
        benchmark="pathorob-camelyon",
        model="RudolfV 2-S",
        inventory_entry=(tmp_path / "cls-only.npy", "reused"),
        canonical_root=tmp_path / "embeddings",
        study_root=tmp_path / "study",
        eval_manifest_root=tmp_path / "manifests",
        device_arg="cpu",
        model_plans=study.POOLING_STUDY_MODELS,
    )

    with config_context(working_memory=777):
        assert study._evaluate_pooling_cell(request) is sentinel
        assert get_config()["working_memory"] == 777

    assert observed_working_memory == [128]


def test_isolated_panel_sends_one_sorted_cell_to_each_runner(tmp_path: Path) -> None:
    requests: list[tuple[str, str, tuple[Path, str]]] = []
    inventory = {
        ("pathorob-tcga-4x4", "RudolfV 2-S"): (tmp_path / "small.npy", "reused"),
        ("pathorob-camelyon", "Mascaret"): (tmp_path / "mascaret.npy", "reused"),
    }

    def fake_cell_runner(request):
        requests.append((request.benchmark, request.model, request.inventory_entry))
        return f"{request.benchmark}/{request.model}"

    runs = study.evaluate_pooling_panel_isolated(
        canonical_root=tmp_path / "canonical",
        study_root=tmp_path / "study",
        eval_manifest_root=tmp_path / "manifests",
        device_arg="cpu",
        inventory=inventory,
        model_plans=study.POOLING_STUDY_MODELS,
        cell_runner=fake_cell_runner,
    )

    assert requests == [
        (
            "pathorob-camelyon",
            "Mascaret",
            (tmp_path / "mascaret.npy", "reused"),
        ),
        (
            "pathorob-tcga-4x4",
            "RudolfV 2-S",
            (tmp_path / "small.npy", "reused"),
        ),
    ]
    assert runs == [
        "pathorob-camelyon/Mascaret",
        "pathorob-tcga-4x4/RudolfV 2-S",
    ]


def test_isolated_panel_default_uses_one_spawned_executor_per_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn_context = object()
    requested_contexts: list[str] = []
    executors = []

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.submissions = []
            executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, function, request):
            self.submissions.append((function, request))
            return FakeFuture(f"{request.benchmark}/{request.model}")

    def fake_get_context(method: str):
        requested_contexts.append(method)
        return spawn_context

    monkeypatch.setattr(study.multiprocessing, "get_context", fake_get_context)
    monkeypatch.setattr(study, "ProcessPoolExecutor", FakeExecutor)
    inventory = {
        ("pathorob-tcga-4x4", "RudolfV 2-S"): (tmp_path / "small.npy", "reused"),
        ("pathorob-camelyon", "Mascaret"): (tmp_path / "mascaret.npy", "reused"),
    }

    runs = study.evaluate_pooling_panel_isolated(
        canonical_root=tmp_path / "canonical",
        study_root=tmp_path / "study",
        eval_manifest_root=tmp_path / "manifests",
        device_arg="cpu",
        inventory=inventory,
        model_plans=study.POOLING_STUDY_MODELS,
    )

    assert requested_contexts == ["spawn", "spawn"]
    assert len(executors) == 2
    assert all(executor.max_workers == 1 for executor in executors)
    assert all(executor.mp_context is spawn_context for executor in executors)
    assert [
        (function, request.benchmark, request.model)
        for executor in executors
        for function, request in executor.submissions
    ] == [
        (study._evaluate_pooling_cell, "pathorob-camelyon", "Mascaret"),
        (study._evaluate_pooling_cell, "pathorob-tcga-4x4", "RudolfV 2-S"),
    ]
    assert runs == [
        "pathorob-camelyon/Mascaret",
        "pathorob-tcga-4x4/RudolfV 2-S",
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


def test_rudolf_cls_only_materializes_the_exact_validated_canonical_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "embeddings"
    study_root = tmp_path / "study"
    directory = canonical_root / "pathorob-camelyon"
    directory.mkdir(parents=True)
    manifest_path = directory / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["a.png", "b.png"],
            "label": ["normal", "tumor"],
            "confounder": ["RUMC", "UMCU"],
            "group_id": ["slide-a", "slide-b"],
        }
    ).to_csv(manifest_path, index=False)
    model = "RudolfV 2-S"
    spec = study._build_model_registry()[model]
    canonical_path = directory / f"{model}.npy"
    canonical = np.arange(2 * 768, dtype=np.float32).reshape(2, 768)
    canonical_contract = study.extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=64,
        device_arg="cpu",
        pooling="canonical",
    )
    study.extraction.publish_embedding_artifact(
        canonical_path,
        canonical,
        canonical_contract,
    )
    protected = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (canonical_path, study.sidecar_path(canonical_path))
    }
    monkeypatch.setattr(
        study.extraction,
        "embed_manifest",
        lambda **kwargs: pytest.fail("Rudolf CLS-only must reuse the validated canonical CLS"),
    )

    target, status = study.extract_study_representation(
        canonical_root=canonical_root,
        study_root=study_root,
        tileset="pathorob-camelyon",
        model=model,
        representation="cls-only",
        batch_size=64,
        num_workers=0,
        device_arg="cpu",
    )

    assert status == "written"
    np.testing.assert_array_equal(np.load(target), canonical[:, :384])
    materialization = json.loads(study.sidecar_path(target).read_text())["extraction_contract"][
        "materialization"
    ]
    assert materialization == {
        "method": "validated-canonical-cls-prefix",
        "output_normalization": "none",
        "slice_start": 0,
        "slice_stop": 384,
        "source_matrix_sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
        "source_representation": "canonical-cls-plus-mean-patches",
        "source_sidecar_sha256": hashlib.sha256(
            study.sidecar_path(canonical_path).read_bytes()
        ).hexdigest(),
        "source_width": 768,
    }
    target_before = (target.read_bytes(), target.stat().st_mtime_ns)
    assert study.extract_study_representation(
        canonical_root=canonical_root,
        study_root=study_root,
        tileset="pathorob-camelyon",
        model=model,
        representation="cls-only",
        batch_size=64,
        num_workers=0,
        device_arg="cpu",
        check=True,
    ) == (target, "checked")
    assert (target.read_bytes(), target.stat().st_mtime_ns) == target_before
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected} == protected


def test_rudolf_cls_derivation_rejects_a_noncanonical_pooling_layout(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "embeddings"
    directory = canonical_root / "pathorob-camelyon"
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
    model = "RudolfV 2-S"
    spec = study._build_model_registry()[model]
    canonical_path = directory / f"{model}.npy"
    wrong_contract = study.extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=64,
        device_arg="cpu",
        pooling="cls-only",
    )
    study.extraction.publish_embedding_artifact(
        canonical_path,
        np.zeros((1, 384), dtype=np.float32),
        wrong_contract,
    )

    with pytest.raises(
        study.ArtifactCompatibilityError,
        match="incompatible embedding artifact provenance: extraction_contract, output_shape",
    ):
        study.extract_study_representation(
            canonical_root=canonical_root,
            study_root=tmp_path / "study",
            tileset="pathorob-camelyon",
            model=model,
            representation="cls-only",
            batch_size=64,
            num_workers=0,
            device_arg="cpu",
        )


def test_alternative_extraction_rejects_a_compatible_nonfinite_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    writes: list[Path] = []

    def fake_embed_manifest(*, output_path, artifact_contract, **kwargs):
        writes.append(Path(output_path))
        study.extraction.publish_embedding_artifact(
            Path(output_path),
            np.zeros((1, 3072), dtype=np.float32),
            artifact_contract,
        )
        return Path(output_path), (1, 3072)

    monkeypatch.setattr(study.extraction, "embed_manifest", fake_embed_manifest)
    assert study.extract_study_representation(
        canonical_root=canonical_root,
        study_root=study_root,
        tileset="pathorob-camelyon",
        model="Mascaret",
        representation="cls-mean-patch",
        batch_size=1,
        num_workers=0,
        device_arg="cpu",
        force=True,
    ) == (target, "forced")
    assert writes == [target]
    assert np.isfinite(np.load(target)).all()


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


def test_rudolf_panel_extraction_inventory_is_exact_across_all_widths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_extract(**kwargs):
        calls.append(kwargs)
        return (
            study.study_embedding_path(
                study_root=kwargs["study_root"],
                tileset=kwargs["tileset"],
                model=kwargs["model"],
                representation=kwargs["representation"],
            ),
            "written",
        )

    monkeypatch.setattr(study, "extract_study_representation", fake_extract)
    inventory = study.extract_pooling_panel(
        canonical_root=tmp_path / "embeddings",
        study_root=tmp_path / "study",
        device_arg="cuda",
        num_workers=4,
        model_plans=study.RUDOLFV2_STUDY_MODELS,
    )

    assert len(inventory) == 12
    assert set(inventory) == {
        (benchmark, model)
        for benchmark in study.PATHOROB_STUDY_BENCHMARKS
        for model in study.RUDOLFV2_STUDY_MODELS
    }
    assert {
        model: {call["batch_size"] for call in calls if call["model"] == model}
        for model in study.RUDOLFV2_STUDY_MODELS
    } == {"RudolfV 2": {32}, "RudolfV 2-B": {32}, "RudolfV 2-S": {64}}
    assert all(call["representation"] == "cls-only" for call in calls)


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


def test_rudolf_inventory_validates_all_twelve_pairs_without_writes(tmp_path: Path) -> None:
    canonical_root = tmp_path / "logical-embeddings"
    study_root = tmp_path / "logical-study"
    canonical_access_root = tmp_path / "access-embeddings"
    study_access_root = tmp_path / "access-study"
    protected: dict[Path, tuple[bytes, int]] = {}
    registry = study._build_model_registry()
    for benchmark in study.PATHOROB_STUDY_BENCHMARKS.values():
        directory = canonical_access_root / benchmark.tileset
        directory.mkdir(parents=True)
        manifest_path = directory / "manifest.csv"
        pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "image_path": ["a.png", "b.png"],
                "label": ["normal", "tumor"],
                "confounder": ["RUMC", "UMCU"],
                "group_id": ["slide-a", "slide-b"],
            }
        ).to_csv(manifest_path, index=False)
        for model, plan in study.RUDOLFV2_STUDY_MODELS.items():
            canonical_path = directory / f"{model}.npy"
            canonical_contract = study.extraction.build_embedding_artifact_contract(
                manifest_path=manifest_path,
                spec=registry[model],
                batch_size=plan.batch_size,
                device_arg="cpu",
                pooling="canonical",
            )
            study.extraction.publish_embedding_artifact(
                canonical_path,
                np.zeros((2, 2 * plan.alternative_width), dtype=np.float32),
                canonical_contract,
            )
            target, status = study.extract_study_representation(
                canonical_root=canonical_access_root,
                study_root=study_access_root,
                tileset=benchmark.tileset,
                model=model,
                representation=plan.alternative,
                batch_size=plan.batch_size,
                num_workers=0,
                device_arg="cpu",
            )
            assert status == "written"
            for path in (target, study.sidecar_path(target)):
                protected[path] = (path.read_bytes(), path.stat().st_mtime_ns)

    inventory = study.build_study_inventory(
        canonical_root=canonical_root,
        study_root=study_root,
        canonical_access_root=canonical_access_root,
        study_access_root=study_access_root,
        model_ids=tuple(study.RUDOLFV2_STUDY_MODELS),
        device_arg="cpu",
        model_plans=study.RUDOLFV2_STUDY_MODELS,
    )

    assert len(inventory) == 12
    assert inventory.groupby("model_registry_id").size().to_dict() == {
        "RudolfV 2": 4,
        "RudolfV 2-B": 4,
        "RudolfV 2-S": 4,
    }
    assert set(inventory["width"]) == {384, 768, 1536}
    assert set(inventory["dtype"]) == {"float32"}
    assert "published_name" not in inventory
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected} == protected

    poisoned = study.study_embedding_path(
        study_root=study_access_root,
        tileset="pathorob-camelyon",
        model="RudolfV 2-S",
        representation="cls-only",
    )
    np.save(poisoned, np.full((2, 384), np.nan, dtype=np.float32))
    with pytest.raises(study.ArtifactCompatibilityError, match="finite FP32"):
        study.build_study_inventory(
            canonical_root=canonical_root,
            study_root=study_root,
            canonical_access_root=canonical_access_root,
            study_access_root=study_access_root,
            model_ids=tuple(study.RUDOLFV2_STUDY_MODELS),
            device_arg="cpu",
            model_plans=study.RUDOLFV2_STUDY_MODELS,
        )


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
