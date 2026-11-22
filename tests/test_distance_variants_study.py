"""The distance-variants study: whitening, the measures, and the cosine reproduction check
with its per-benchmark tolerance.

Paper-free on purpose (it stays in the sdist): nothing here imports ``scripts/repro`` or
reads ``output/``. The table and macros built from the study are tested in
``test_distance_variants_table.py``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "scripts" / "studies", ROOT / "scripts" / "bench"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import distance_variants as dv  # noqa: E402

#: The study's own Manhattan distance, which needs CUDA; every other test scores Manhattan
#: cells through :func:`_manhattan_on_the_cpu` instead.
_GPU_MANHATTAN = dv.manhattan_distance
#: What the CPU stand-in records as its computation.
_CPU_COMPUTATION = {"backend": "numpy (test stand-in)", "dtype": "float64"}


def _numpy_l1(rows: np.ndarray) -> np.ndarray:
    return np.abs(rows[:, None, :] - rows[None, :, :]).sum(axis=2)


@pytest.fixture(autouse=True)
def _manhattan_on_the_cpu(monkeypatch):
    monkeypatch.setattr(dv, "manhattan_distance", lambda: (_numpy_l1, dict(_CPU_COMPUTATION)))


# --------------------------------------------------------------------------------------
# Whitening.
# --------------------------------------------------------------------------------------

#: Population variances along six orthogonal directions; cumulative shares 0.50, 0.80,
#: 0.94, 0.98, 0.995, 1.0. So 90% needs three components, 95% four and 99% five.
_SPECTRUM = np.array([50.0, 30.0, 14.0, 4.0, 1.5, 0.5])


def _known_spectrum(n: int = 400, seed: int = 3) -> np.ndarray:
    """Rows whose *sample* covariance is exactly ``R diag(_SPECTRUM) R^T`` for a rotation R.

    Centred orthonormal columns scaled by ``sqrt(n - 1)`` have identity sample covariance,
    so the spectrum is the sample spectrum, not merely its expectation.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, len(_SPECTRUM)))
    q, _ = np.linalg.qr(raw - raw.mean(axis=0))
    rotation, _ = np.linalg.qr(rng.standard_normal((len(_SPECTRUM),) * 2))
    return 7.0 + np.sqrt(n - 1) * (q * np.sqrt(_SPECTRUM)) @ rotation.T


@pytest.mark.parametrize("variance, expected", [(0.90, 3), (0.95, 4), (0.99, 5)])
def test_whitening_keeps_the_fewest_components_reaching_the_variance_share(variance, expected):
    whitened = dv.whiten(_known_spectrum(), variance)

    assert whitened.n_components == expected
    assert whitened.explained == pytest.approx(np.cumsum(_SPECTRUM)[expected - 1] / 100.0)
    assert whitened.features.shape == (400, expected)


def test_whitened_components_have_identity_covariance():
    whitened = dv.whiten(_known_spectrum(), 0.95)

    covariance = np.cov(whitened.features, rowvar=False, ddof=1)
    np.testing.assert_allclose(covariance, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(whitened.features.mean(axis=0), 0.0, atol=1e-10)


def test_a_tile_scored_in_several_subsets_is_whitened_once_not_weighted_by_its_count():
    tiles = _known_spectrum(n=60)
    occurrences = np.r_[np.arange(60), np.zeros(40, dtype=int)]  # tile 0 occurs 41 times
    variant = dv.variant("whitened")

    once, _ = dv.prepare(tiles, np.arange(60), variant)
    repeated, record = dv.prepare(tiles[occurrences], occurrences, variant)

    np.testing.assert_allclose(repeated, once[occurrences], atol=1e-12)
    assert record["whitening_rows"] == 60
    assert record["n_components"] == 4


def test_normalised_euclidean_scores_unit_rows():
    rows = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    prepared, _ = dv.prepare(rows, np.arange(2), dv.variant("normalised-euclidean"))

    np.testing.assert_allclose(prepared, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)


# --------------------------------------------------------------------------------------
# The measures.
# --------------------------------------------------------------------------------------


def _summary() -> pd.DataFrame:
    """Four ranked encoders and the control, scored under cosine and one variant.

    By hand: the variant keeps the median ranking (rho = 1) and reverses the LTM10 one
    (rho = -1); E2's median crosses zero (0.10 -> -0.05), and so does the control's,
    which must not count; |delta F(0)| is 0.05, 0.10, 0, 0.20, median 0.075; the
    per-sample sign agreements 0.9, 0.8, 0.95, 0.7 have median 0.85.
    """
    rows = []
    cosine = {
        "E1": (0.30, 0.10, -0.40, 1.0),
        "E2": (0.10, 0.20, -0.30, 1.0),
        "E3": (-0.10, 0.60, -0.20, 1.0),
        "E4": (-0.20, 0.70, -0.10, 1.0),
        dv.CONTROL_MODEL: (0.05, 0.40, -0.50, 1.0),
    }
    other = {
        "E1": (0.25, 0.15, -0.10, 0.90),
        "E2": (-0.05, 0.30, -0.20, 0.80),
        "E3": (-0.15, 0.60, -0.30, 0.95),
        "E4": (-0.30, 0.50, -0.40, 0.70),
        dv.CONTROL_MODEL: (-0.10, 0.60, -0.90, 0.10),
    }
    for name, table in (("cosine", cosine), ("euclidean", other)):
        for model, (median, f0, ltm10, agreement) in table.items():
            rows.append(
                {
                    "benchmark": "toy",
                    "model": model,
                    "variant": name,
                    "median": median,
                    "f0": f0,
                    "ltm10": ltm10,
                    "sign_agreement": agreement,
                    "n_components": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _row(table: pd.DataFrame, variant: str) -> pd.Series:
    return table.set_index("variant").loc[variant]


def test_measures_compare_each_variant_with_cosine_over_the_ranked_encoders():
    row = _row(dv.measures(_summary()), "euclidean")

    assert row["n_encoders"] == 4
    assert row["rank_rho_median"] == pytest.approx(1.0)
    assert (row["rank_rho_median_lo"], row["rank_rho_median_hi"]) == pytest.approx((1.0, 1.0))
    assert row["rank_rho_ltm10"] == pytest.approx(-1.0)
    assert row["sign_changes"] == 1
    assert row["median_abs_delta_f0"] == pytest.approx(0.075)
    assert row["median_sign_agreement"] == pytest.approx(0.85)


def test_the_nipd_association_is_reported_beside_cosines_with_an_exact_permutation_p():
    nipd = pd.DataFrame(
        {
            "benchmark": ["toy"] * 5,
            "model": ["E1", "E2", "E3", "E4", dv.CONTROL_MODEL],
            "nipd_id": [-0.01, -0.02, -0.03, -0.04, -0.5],
            "nipd_ood": [-0.04, -0.03, -0.02, -0.01, 0.5],
        }
    )

    table = dv.measures(_summary(), nipd)

    for variant in ("cosine", "euclidean"):
        row = _row(table, variant)
        assert row["rho_nipd_id"] == pytest.approx(1.0)
        assert row["rho_nipd_ood"] == pytest.approx(-1.0)
        # Four encoders: all 24 orderings, of which the identity and the reversal reach |rho| = 1.
        assert row["rho_nipd_id_p"] == pytest.approx(2 / 24)
    assert np.isnan(_row(table, "cosine")["rank_rho_median"])


def test_sign_agreement_reads_zero_as_confounder_dominant_as_f0_does():
    assert dv.sign_agreement([0.0, 0.2, -0.1, 0.3], [-0.2, 0.0, -0.3, 0.1]) == 0.75


# --------------------------------------------------------------------------------------
# The reproduction tolerance.
# --------------------------------------------------------------------------------------

TOLKACH = "pathorob-tolkach-esca"
EXACT_BENCHMARKS = ("pathorob-camelyon", "pathorob-tcga-2x2", "pathorob-tcga-4x4")


def _attempt(deviation: float, n_mismatched: int = 1) -> dict:
    """A cosine attempt record, as far as choosing a kernel reads it."""
    identical = deviation == 0.0
    return {
        "reproduction": {
            "identical": identical,
            "max_abs_deviation": deviation,
            "n_mismatched": 0 if identical else n_mismatched,
        }
    }


#: UNI2-h on Tolkach-ESCA: neither kernel reproduces the run exactly, SkylakeX deviates least.
_UNI2H = {"Cooperlake": _attempt(4.636704759031218e-07, 8854), "SkylakeX": _attempt(1.36e-08)}


def test_only_tolkach_esca_is_checked_within_a_tolerance():
    assert set(dv.REPRODUCTION_TOLERANCE) == set(dv.BENCHMARKS)
    assert {b for b, t in dv.REPRODUCTION_TOLERANCE.items() if t > 0} == {TOLKACH}
    assert dv.reproduction_tolerance(TOLKACH) == 1e-4


def test_within_tolkachs_tolerance_the_kernel_that_deviates_least_is_accepted():
    choice = dv.choose_kernel(_UNI2H, dv.reproduction_tolerance(TOLKACH))

    assert choice == dv.Assignment("SkylakeX", 1.36e-08)


def test_an_exact_kernel_deviates_least_and_the_first_of_equals_wins():
    tolerance = dv.reproduction_tolerance(TOLKACH)
    exact_second = {"Cooperlake": _attempt(1e-9), "SkylakeX": _attempt(0.0)}
    both_exact = {"Cooperlake": _attempt(0.0), "SkylakeX": _attempt(0.0)}

    assert dv.choose_kernel(exact_second, tolerance) == dv.Assignment("SkylakeX", 0.0)
    assert dv.choose_kernel(both_exact, 0.0) == dv.Assignment("Cooperlake", 0.0)


@pytest.mark.parametrize("benchmark", EXACT_BENCHMARKS)
def test_the_same_deviation_on_an_exact_benchmark_is_refused(benchmark):
    assert dv.choose_kernel(_UNI2H, dv.reproduction_tolerance(benchmark)) is None


def test_a_deviation_above_the_tolerance_is_refused():
    above = {"Cooperlake": _attempt(2.0e-4), "SkylakeX": _attempt(1.5e-4)}

    assert dv.choose_kernel(above, dv.reproduction_tolerance(TOLKACH)) is None


def test_a_record_passes_only_within_its_benchmarks_tolerance_as_it_stands():
    exact = {"reproduction": {"identical": True, "max_abs_deviation": 0.0}}
    within = {"reproduction": {"identical": False}, "max_deviation": 1.36e-08}
    above = {"reproduction": {"identical": False}, "max_deviation": 2.0e-4}
    # A record no tolerance accepted (it carries no max_deviation) never passes.
    unaccepted = {"reproduction": {"identical": False, "max_abs_deviation": 1.36e-08}}

    assert dv.reproduction_passed(exact, "pathorob-camelyon")
    assert dv.reproduction_passed(exact, TOLKACH)
    assert dv.reproduction_passed(within, TOLKACH)
    assert not dv.reproduction_passed(within, "pathorob-camelyon")
    assert not dv.reproduction_passed(above, TOLKACH)
    assert not dv.reproduction_passed(unaccepted, TOLKACH)


# --------------------------------------------------------------------------------------
# One encoder x benchmark, end to end against a real driver run.
# --------------------------------------------------------------------------------------

_CELLS = [(label, centre) for label in ("A", "B") for centre in ("C1", "C2")]


def _toy_run(bench_env):
    """A 4 x 10-tile benchmark scored by the driver at m up to 5, as a production run is."""
    rng = np.random.default_rng(5)
    rows, labels, centres = [], [], []
    axes = {key: rng.standard_normal(12) for key in ("A", "B", "C1", "C2")}
    for label, centre in _CELLS:
        for _ in range(10):
            rows.append(axes[label] + 0.8 * axes[centre] + rng.standard_normal(12))
            labels.append(label)
            centres.append(centre)
    n = len(rows)
    tileset = pd.DataFrame(
        {
            "sample_id": [f"t{i}" for i in range(n)],
            "image_path": [f"/tmp/t{i}.png" for i in range(n)],
            "label": labels,
            "medical_center": centres,
            "group_id": [f"slide{i}" for i in range(n)],
        }
    )
    bench_env.write_tileset("toy-tiles", tileset, {"M1": np.asarray(rows)})
    bench_env.register("toy", tileset="toy-tiles", manifest=tileset.assign(dataset="toy"), k_max=5)
    assert bench_env.run("toy", "k-star", "--croma-m-max", "5", "--progress", "off") == 0
    results_dir = bench_env.results_dir("toy")
    run = dv.Run(
        benchmark="toy",
        design="all",
        results_dir=results_dir,
        search=dv.Search.from_run_config(results_dir / "run_config.json"),
    )
    from views import load_view

    return run, load_view("toy")


#: The embedding digest ``_run_unit`` records in every cell.
_SCORED_FROM = "0" * 64


def _run_unit(root, run, view, *, kernel="test", within_tolerance=False):
    return dv.run_unit(
        root,
        run,
        "M1",
        features=view.features("M1"),
        rows=view.rows,
        manifest=view.eval_manifest,
        kernel=kernel,
        within_tolerance=within_tolerance,
        embedding_sha256=_SCORED_FROM,
    )


def _shift_stored_value(run, by: float, row: int = 3) -> None:
    """Move one of the run's stored per-sample values by ``by``."""
    stored = pd.read_csv(run.per_sample_csv("M1"), float_precision="round_trip")
    stored.loc[row, "croma_m5"] = stored.loc[row, "croma_m5"] + by
    stored.to_csv(run.per_sample_csv("M1"), index=False)


def _toy_design(run) -> dict:
    return {
        "benchmarks": {
            "toy": {
                "results_dir": str(run.results_dir),
                "models": ["M1"],
                "variants": [{"name": v.name} for v in dv.VARIANTS],
            }
        }
    }


def test_the_cosine_variant_reproduces_the_runs_stored_values_bit_for_bit(bench_env, tmp_path):
    run, view = _toy_run(bench_env)
    root = tmp_path / "study"

    assert _run_unit(root, run, view) == 0

    cosine = dv.read_cell(root, "toy", "M1", "cosine")
    assert cosine["reproduced"] is True
    assert all(cosine["reproduction"]["pooled_identical"].values())
    stored = pd.read_csv(run.per_sample_csv("M1"), float_precision="round_trip")
    values = dv.read_cell_values(root, "toy", "M1", "cosine")
    assert values["sample_id"].tolist() == stored["sample_id"].astype(str).tolist()
    assert values["croma"].tolist() == stored["croma_m5"].tolist()
    for name in ("normalised-euclidean", "euclidean", "whitened", "normalised-manhattan"):
        cell = dv.read_cell(root, "toy", "M1", name)
        assert cell["n_units"] == 40 and cell["kernel"] == "test"
    assert 1 <= dv.read_cell(root, "toy", "M1", "whitened")["n_components"] <= 12


def test_normalised_manhattan_scores_manhattan_distance_on_the_unit_rows(bench_env, tmp_path):
    run, view = _toy_run(bench_env)
    root = tmp_path / "study"

    assert _run_unit(root, run, view) == 0

    # Brute force on the L2-normalised rows, at the headline m: every tile is its own slide,
    # so no candidate is excluded, and the toy run's search reaches every typed neighbour.
    features = np.asarray(view.features("M1"), dtype=np.float64)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    manifest = view.eval_manifest
    labels, centres = manifest["label"].to_numpy(), manifest["confounder"].to_numpy()
    dist = np.abs(unit[:, None, :] - unit[None, :, :]).sum(axis=2)
    expected = {}
    for i, sample_id in enumerate(manifest["sample_id"].astype(str)):
        others = np.arange(len(unit)) != i
        d_so = np.sort(dist[i, others & (labels == labels[i]) & (centres != centres[i])])[:5]
        d_os = np.sort(dist[i, others & (labels != labels[i]) & (centres == centres[i])])[:5]
        expected[sample_id] = (d_os.mean() - d_so.mean()) / (d_os.mean() + d_so.mean())
    cell = dv.read_cell(root, "toy", "M1", "normalised-manhattan")
    values = dv.read_cell_values(root, "toy", "M1", "normalised-manhattan")

    assert (cell["distance"], cell["feature_preparation"], cell["m"]) == ("manhattan", "unit", 5)
    assert "n_components" not in cell  # nothing is fitted
    assert cell["distance_computation"] == _CPU_COMPUTATION
    np.testing.assert_allclose(
        values["croma"], [expected[s] for s in values["sample_id"]], rtol=0, atol=1e-6
    )


def test_only_the_manhattan_cell_records_a_distance_computed_outside_the_library(
    bench_env, tmp_path
):
    run, view = _toy_run(bench_env)
    root = tmp_path / "study"

    assert _run_unit(root, run, view) == 0

    for v in dv.VARIANTS:
        cell = dv.read_cell(root, "toy", "M1", v.name)
        assert ("distance_computation" in cell) == (v.distance == "manhattan"), v.name


# --------------------------------------------------------------------------------------
# Manhattan distances on the GPU.
# --------------------------------------------------------------------------------------


def test_manhattan_cells_are_not_scored_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(RuntimeError, match="CUDA"):
        _GPU_MANHATTAN()


def test_manhattan_cells_are_not_scored_on_the_cpu_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*CPU"):
        _GPU_MANHATTAN()


def test_a_sweep_with_manhattan_cells_to_score_stops_before_starting_a_unit_without_cuda(
    bench_env, tmp_path, monkeypatch
):
    run, _ = _toy_run(bench_env)
    monkeypatch.setattr(dv, "manhattan_distance", _GPU_MANHATTAN)
    monkeypatch.setitem(sys.modules, "torch", None)
    started = []
    monkeypatch.setattr(dv.subprocess, "Popen", lambda *a, **k: started.append(a))

    with pytest.raises(RuntimeError, match="CUDA"):
        dv.sweep(tmp_path / "study", _toy_design(run), workers=1)
    assert started == []


def test_torch_manhattan_distances_are_the_float64_sums_in_every_query_chunk(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(dv, "L1_QUERY_CHUNK", 3)
    # Unit float32 rows, as the study scores: their L1 sums are not exact in float32, so a
    # float32 computation would miss the float64 sums by far more than atol.
    gaussian = np.random.default_rng(11).standard_normal((10, 16))
    rows = (gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)).astype(np.float32)

    distances = dv.torch_l1_distances(rows, device="cpu")

    assert distances.dtype == np.float64
    np.testing.assert_allclose(distances, _numpy_l1(rows.astype(np.float64)), rtol=0, atol=1e-12)
    assert np.all(np.diagonal(distances) == 0.0)


def test_the_cuda_manhattan_distance_scores_exactly_as_the_manhattan_distance():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from croma import CRoMa
    from metric_harness import CONFOUNDER_COLUMN, toy_manifest

    # Distinct small-integer rows: every L1 distance is an exact sum on either device, and
    # many tie, so the two agree only if the search keeps every neighbour and tie the same.
    rows = np.random.default_rng(5).integers(-2, 3, size=(36, 6)).astype(np.float64)
    assert len({tuple(row) for row in rows}) == len(rows)
    manifest = toy_manifest(["A", "B"] * 18, [f"C{i % 3}" for i in range(36)])
    distance, computation = _GPU_MANHATTAN()
    assert computation["backend"] == "torch.cdist(p=1)"

    def croma(distance):
        return CRoMa.compute(
            rows, manifest, confounder_column=CONFOUNDER_COLUMN, m=5, distance=distance
        )

    on_cuda, reference = croma(distance), croma("manhattan")

    assert np.array_equal(on_cuda.sample_values_aligned, reference.sample_values_aligned)
    assert on_cuda.value == reference.value


def test_a_unit_within_tolkachs_tolerance_is_scored_under_the_kernel_that_deviates_least(
    bench_env, tmp_path, monkeypatch
):
    run, view = _toy_run(bench_env)
    monkeypatch.setitem(dv.REPRODUCTION_TOLERANCE, "toy", dv.reproduction_tolerance(TOLKACH))
    _shift_stored_value(run, 2e-8)
    root = tmp_path / "study"

    # Not exact under SkylakeX: an attempt is recorded, and no cell.
    assert _run_unit(root, run, view, kernel="SkylakeX") == dv.NOT_REPRODUCED
    assert dv.read_cell(root, "toy", "M1", "cosine") is None
    attempt = dv.read_attempts(root, "toy", "M1")["SkylakeX"]
    deviation = attempt["reproduction"]["max_abs_deviation"]
    assert deviation == pytest.approx(2e-8, rel=1e-6)
    # Accepting a kernel within tolerance needs every candidate tried.
    with pytest.raises(RuntimeError, match="Cooperlake"):
        _run_unit(root, run, view, kernel="SkylakeX", within_tolerance=True)

    # One process has one kernel, so Cooperlake's attempt is written by hand.
    def cooperlake_deviates(by: float) -> None:
        record = json.loads(json.dumps(attempt))
        record.update(kernel="Cooperlake")
        record["reproduction"].update(max_abs_deviation=by, n_mismatched=38)
        dv.attempt_path(root, "toy", "M1", "Cooperlake").write_text(json.dumps(record))

    # SkylakeX is refused while Cooperlake deviates less ...
    cooperlake_deviates(1e-9)
    assert _run_unit(root, run, view, kernel="SkylakeX", within_tolerance=True) == (
        dv.NOT_REPRODUCED
    )
    assert dv.read_cell(root, "toy", "M1", "cosine") is None
    # ... and accepted once it deviates least, as on UNI2-h.
    cooperlake_deviates(4.6e-7)
    assert dv.choose_kernel(dv.read_attempts(root, "toy", "M1"), 1e-4).kernel == "SkylakeX"
    assert _run_unit(root, run, view, kernel="SkylakeX", within_tolerance=True) == 0

    cosine = dv.read_cell(root, "toy", "M1", "cosine")
    assert cosine["kernel"] == "SkylakeX" and cosine["reproduced"] is True
    assert cosine["reproduction"]["identical"] is False
    assert cosine["reproduction"]["n_mismatched"] == 1
    assert (cosine["tolerance"], cosine["max_deviation"]) == (1e-4, deviation)
    assert cosine["kernel_deviations"] == {"Cooperlake": 4.6e-7, "SkylakeX": deviation}
    for name in ("normalised-euclidean", "euclidean", "whitened"):
        assert dv.read_cell(root, "toy", "M1", name)["kernel"] == "SkylakeX"

    design = _toy_design(run)
    assert dv.reproduction_summary(root, design) == {
        "toy": {
            "tolerance": 1e-4,
            "identical": False,
            "max_deviation": deviation,
            "within_tolerance": {
                "M1": {"kernel": "SkylakeX", "max_deviation": deviation, "n_mismatched": 1}
            },
        }
    }
    # Tightening the tolerance invalidates the record it no longer covers.
    monkeypatch.setitem(dv.REPRODUCTION_TOLERANCE, "toy", 0.0)
    (root / dv.DESIGN).write_text("{}")
    with pytest.raises(dv.IncompleteStudyError, match="does not reproduce"):
        dv.summarise(root, design, apd_correlation_csv=None)


@pytest.mark.parametrize(
    "tolerance_of, shift",
    [("pathorob-camelyon", 2e-8), (TOLKACH, 2e-4)],
    ids=["same-deviation-on-an-exact-benchmark", "above-tolkachs-tolerance"],
)
def test_a_unit_outside_its_tolerance_is_refused(
    bench_env, tmp_path, monkeypatch, tolerance_of, shift
):
    run, view = _toy_run(bench_env)
    monkeypatch.setitem(dv.REPRODUCTION_TOLERANCE, "toy", dv.reproduction_tolerance(tolerance_of))
    _shift_stored_value(run, shift)
    root = tmp_path / "study"

    for kernel in dv.CANDIDATE_KERNELS:
        assert _run_unit(root, run, view, kernel=kernel) == dv.NOT_REPRODUCED
    attempts = dv.read_attempts(root, "toy", "M1")
    assert dv.choose_kernel(attempts, dv.reproduction_tolerance("toy")) is None
    for kernel in dv.CANDIDATE_KERNELS:
        assert _run_unit(root, run, view, kernel=kernel, within_tolerance=True) == (
            dv.NOT_REPRODUCED
        )

    assert dv.read_cell(root, "toy", "M1", "cosine") is None
    (root / dv.DESIGN).write_text("{}")
    with pytest.raises(dv.IncompleteStudyError, match="cells missing"):
        dv.summarise(root, _toy_design(run), apd_correlation_csv=None)


# --------------------------------------------------------------------------------------
# A finished study extended by a new variant.
# --------------------------------------------------------------------------------------

_NEW = "normalised-manhattan"


def _design_record(variants, models=("M1", "M2")) -> dict:
    """A design as ``dv.design`` writes it, for one benchmark scored under ``variants``."""
    spec = {"design": "all", "models": list(models), "variants": [asdict(v) for v in variants]}
    return {"m": 5, "seed": dv.SEED, "benchmarks": {"toy": spec}}


#: A study written before the Manhattan variant, with a sensitivity variant after the headline
#: ones (as Camelyon's whitening thresholds are), and the same study with it added.
_BEFORE = [v for v in dv.VARIANTS if v.name != _NEW] + [dv.variant("whitened-90")]
_AFTER = list(dv.VARIANTS) + [dv.variant("whitened-90")]


def test_a_design_that_only_adds_a_variant_extends_the_stored_study(tmp_path):
    dv.write_design(tmp_path, _design_record(_BEFORE))

    dv.write_design(tmp_path, _design_record(_AFTER))

    assert json.loads((tmp_path / dv.DESIGN).read_text()) == _design_record(_AFTER)


@pytest.mark.parametrize(
    "record",
    [
        _design_record(_AFTER, models=("M1", "M3")),
        _design_record([v for v in _AFTER if v.name != "whitened"]),
        _design_record(
            [
                dv.Variant("whitened", "whitened", "euclidean", 0.90) if v.name == "whitened" else v
                for v in _AFTER
            ]
        ),
    ],
    ids=["roster-changed", "variant-dropped", "variant-redefined"],
)
def test_a_design_that_changes_anything_but_adding_variants_is_refused(tmp_path, record):
    dv.write_design(tmp_path, _design_record(_BEFORE))
    stored = (tmp_path / dv.DESIGN).read_bytes()

    with pytest.raises(RuntimeError, match="different study"):
        dv.write_design(tmp_path, record)

    assert (tmp_path / dv.DESIGN).read_bytes() == stored


def test_a_finished_unit_gains_the_new_variant_without_rewriting_its_cells(bench_env, tmp_path):
    run, view = _toy_run(bench_env)
    root = tmp_path / "study"
    assert _run_unit(root, run, view) == 0
    # As a unit finished before the variant existed: every cell but the new one.
    for path in dv.cell_paths(root, "toy", "M1", _NEW):
        path.unlink()
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in dv.cell_dir(root, "toy", "M1").iterdir()
    }

    assert _run_unit(root, run, view) == 0

    after = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in dv.cell_dir(root, "toy", "M1").iterdir()
    }
    assert set(after) - set(before) == {f"{_NEW}.json", f"{_NEW}.csv.gz"}
    assert {name: after[name] for name in before} == before


# --------------------------------------------------------------------------------------
# The embeddings a unit's cells were scored from.
# --------------------------------------------------------------------------------------


def test_a_unit_whose_cosine_cell_was_scored_from_other_embeddings_is_not_extended(
    bench_env, tmp_path
):
    run, view = _toy_run(bench_env)
    root = tmp_path / "study"
    assert _run_unit(root, run, view) == 0
    euclidean, _ = dv.cell_paths(root, "toy", "M1", "euclidean")
    euclidean.unlink()

    # Regenerated in place: the cosine cell on disk no longer describes these features.
    with pytest.raises(RuntimeError, match="embeddings"):
        dv.run_unit(
            root,
            run,
            "M1",
            features=view.features("M1"),
            rows=view.rows,
            manifest=view.eval_manifest,
            kernel="test",
            embedding_sha256="1" * 64,
        )

    assert not euclidean.exists()


def _finished_toy_study(bench_env, tmp_path, monkeypatch) -> tuple[Path, dict]:
    """Every cell of the toy unit, reproduced exactly; ready to summarise."""
    run, view = _toy_run(bench_env)
    monkeypatch.setitem(dv.REPRODUCTION_TOLERANCE, "toy", 0.0)
    root = tmp_path / "study"
    assert _run_unit(root, run, view) == 0
    (root / dv.DESIGN).write_text("{}")
    return root, _toy_design(run)


def test_a_study_scored_from_embeddings_that_have_since_changed_is_refused(
    bench_env, tmp_path, monkeypatch
):
    root, design = _finished_toy_study(bench_env, tmp_path, monkeypatch)

    with pytest.raises(dv.IncompleteStudyError, match="toy / M1: the embeddings on disk"):
        dv.summarise(root, design, apd_correlation_csv=None, embedding_digest=lambda b, m: "1" * 64)

    assert not (root / dv.PROVENANCE).exists()


def test_a_unit_whose_cells_were_scored_from_different_embeddings_is_refused(
    bench_env, tmp_path, monkeypatch
):
    root, design = _finished_toy_study(bench_env, tmp_path, monkeypatch)
    whitened, _ = dv.cell_paths(root, "toy", "M1", "whitened")
    record = json.loads(whitened.read_text())
    whitened.write_text(json.dumps({**record, "embedding_sha256": "1" * 64}))

    # The digest on disk matches the cosine cell's, so only the mix gives the unit away.
    with pytest.raises(dv.IncompleteStudyError, match="toy / M1: .*different embeddings"):
        dv.summarise(
            root, design, apd_correlation_csv=None, embedding_digest=lambda b, m: _SCORED_FROM
        )

    assert not (root / dv.PROVENANCE).exists()


# --------------------------------------------------------------------------------------
# The run's per-sample file each cosine cell was checked against.
# --------------------------------------------------------------------------------------


def test_a_study_whose_runs_per_sample_file_has_since_been_rewritten_is_refused(
    bench_env, tmp_path, monkeypatch
):
    root, design = _finished_toy_study(bench_env, tmp_path, monkeypatch)
    results_dir = Path(design["benchmarks"]["toy"]["results_dir"])
    metrics_before = (results_dir / "metrics.csv").read_bytes()
    reference = dv.read_cell(root, "toy", "M1", "cosine")["reproduction"]["reference"]
    # Re-scored within a tolerance, say: the pooled scores in metrics.csv do not move, but
    # the per-sample file the cosine cell reproduced is no longer the one on disk.
    stored = pd.read_csv(reference, float_precision="round_trip")
    stored.loc[3, "croma_m5"] = stored.loc[3, "croma_m5"] + 1e-9
    stored.to_csv(reference, index=False)
    assert (results_dir / "metrics.csv").read_bytes() == metrics_before

    with pytest.raises(dv.IncompleteStudyError, match="toy / M1: .*per-sample.*re-run the sweep"):
        dv.summarise(
            root, design, apd_correlation_csv=None, embedding_digest=lambda b, m: _SCORED_FROM
        )

    assert not (root / dv.PROVENANCE).exists()


# --------------------------------------------------------------------------------------
# The nIPD association, checked against tab:apd-correlation.
# --------------------------------------------------------------------------------------


def _apd_correlation(path: Path, rho_id: float, rho_ood: float, n: int) -> Path:
    pd.DataFrame(
        {
            "target": list(dv.NIPD_TARGETS),
            "metric": ["croma", "croma"],
            "scope": ["camelyon", "camelyon"],
            "spearman": [rho_id, rho_ood],
            "n": [n, n],
        }
    ).to_csv(path, index=False)
    return path


def _cosine_measures(rho_id: float, rho_ood: float, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benchmark": ["pathorob-camelyon"],
            "variant": ["cosine"],
            "n_encoders": [n],
            "rho_nipd_id": [rho_id],
            "rho_nipd_ood": [rho_ood],
        }
    )


def test_cosines_nipd_association_must_equal_the_apd_correlation_table(tmp_path):
    corr = _apd_correlation(tmp_path / "apd_correlation.csv", 0.95, 0.74, 26)

    dv.check_against_apd_correlation(_cosine_measures(0.95, 0.74, 26), corr)
    with pytest.raises(ValueError, match="nipd_ood"):
        dv.check_against_apd_correlation(_cosine_measures(0.95, 0.70, 26), corr)


def test_a_cohort_whose_nipd_association_is_nan_is_refused(tmp_path):
    corr = _apd_correlation(tmp_path / "apd_correlation.csv", 0.95, 0.74, 26)

    with pytest.raises(ValueError, match="pathorob-camelyon: cosine rho\\(CRoMa, nipd_id\\) = nan"):
        dv.check_against_apd_correlation(_cosine_measures(np.nan, 0.74, 26), corr)


def test_an_nipd_cohort_missing_from_the_apd_study_is_refused(tmp_path):
    apd = tmp_path / "apd.csv"
    pd.DataFrame(
        {
            "dataset": ["camelyon", "tcga_4x4"],
            "model": ["E1", "E1"],
            "nipd_id": [-0.01, -0.02],
            "nipd_ood": [-0.03, -0.04],
        }
    ).to_csv(apd, index=False)

    with pytest.raises(ValueError, match="tolkach"):
        dv.load_nipd(apd)
