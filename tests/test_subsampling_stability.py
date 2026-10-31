"""The cohort-subsampling stability study: its design, its replicates and its measures.

The study re-runs the production benchmark driver on subsamples of each tile cohort and
asks how far RI, MaRI and CRoMa scores and encoder rankings move. These tests pin the three
things a reader has to trust before reading any of its numbers: that every replicate keeps
the stated number of units in every class x centre cell, that one draw is shared by every
encoder and metric, and that the 100% "subsample" is the production run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts" / "studies", ROOT / "scripts" / "bench"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import subsampling_stability as study  # noqa: E402


def _cohort_manifest(
    *,
    classes: int,
    centres: int,
    units_per_cell: int,
    tiles_per_unit: int,
    shared_cases: bool = False,
) -> pd.DataFrame:
    """A manifest with the cohort's cell structure: classes x centres, equal blocks per cell.

    ``shared_cases`` reproduces Tolkach-ESCA's shape, where one case contributes a block to
    several classes, so a case is not a unit -- its block within a cell is.
    """
    rows = []
    for c in range(centres):
        for y in range(classes):
            for u in range(units_per_cell):
                case = f"case{c}_{u}" if shared_cases else f"slide{c}_{y}_{u}"
                for t in range(tiles_per_unit):
                    rows.append(
                        {
                            "sample_id": f"{case}__y{y}__t{t}",
                            "image_path": f"/tiles/{case}__y{y}__t{t}.png",
                            "label": f"class{y}",
                            "medical_center": f"centre{c}",
                            "group_id": case,
                        }
                    )
    return pd.DataFrame(rows)


def _units_per_cell(manifest: pd.DataFrame) -> pd.Series:
    return manifest.groupby(["label", "medical_center"])["group_id"].nunique()


# The issue's design table: full units per cell, then the stated count at 25/50/75%.
STATED_DESIGN = {
    "pathorob-camelyon": (17, {25: 4, 50: 8, 75: 13}),
    "pathorob-tcga-4x4": (12, {25: 3, 50: 6, 75: 9}),
    "pathorob-tolkach-esca": (5, {25: 2, 50: 3, 75: 4}),
}


class TestSubsamplingDesign:
    @pytest.mark.parametrize("cohort", sorted(STATED_DESIGN))
    def test_every_cell_keeps_the_stated_unit_count_in_every_replicate(self, cohort):
        full, stated = STATED_DESIGN[cohort]
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=full, tiles_per_unit=3)
        for fraction, expected in stated.items():
            for replicate in range(study.N_REPLICATES):
                drawn = study.draw_subsample(manifest, cohort, fraction, replicate)
                counts = _units_per_cell(drawn)
                assert len(counts) == 4
                assert set(counts) == {expected}, (cohort, fraction, replicate)

    def test_a_unit_is_a_case_block_within_a_cell_and_is_kept_whole(self):
        # Tolkach-ESCA: each case contributes a 3-tile block to every class it spans.
        manifest = _cohort_manifest(
            classes=3, centres=2, units_per_cell=5, tiles_per_unit=3, shared_cases=True
        )
        split_cases = 0
        for replicate in range(study.N_REPLICATES):
            drawn = study.draw_subsample(manifest, "pathorob-tolkach-esca", 50, replicate)
            blocks = drawn.groupby(["label", "medical_center", "group_id"]).size()
            assert set(blocks) == {3}, "a sampled block lost tiles"
            # Sampling blocks, not cases: some case keeps a block in one class and loses
            # its block in another.
            classes_kept = drawn.groupby("group_id")["label"].nunique()
            split_cases += int((classes_kept < 3).sum())
        assert split_cases > 0

    def test_the_full_fraction_is_the_production_manifest_unchanged(self):
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=17, tiles_per_unit=3)
        shuffled = manifest.sample(frac=1.0, random_state=0).reset_index(drop=True)
        drawn = study.draw_subsample(shuffled, "pathorob-camelyon", 100, 0)
        pd.testing.assert_frame_equal(drawn, shuffled)

    def test_a_subsample_keeps_the_manifest_row_order(self):
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=17, tiles_per_unit=3)
        shuffled = manifest.sample(frac=1.0, random_state=1).reset_index(drop=True)
        drawn = study.draw_subsample(shuffled, "pathorob-camelyon", 50, 3)
        positions = shuffled.reset_index().merge(drawn, on=list(shuffled.columns))
        assert positions["index"].is_monotonic_increasing
        assert list(drawn.columns) == list(shuffled.columns)

    def test_a_replicate_is_one_fixed_draw_and_replicates_differ(self):
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=12, tiles_per_unit=2)
        first = study.draw_subsample(manifest, "pathorob-tcga-4x4", 25, 7)
        again = study.draw_subsample(manifest.copy(), "pathorob-tcga-4x4", 25, 7)
        pd.testing.assert_frame_equal(first, again)
        draws = {
            tuple(study.draw_subsample(manifest, "pathorob-tcga-4x4", 25, r)["sample_id"])
            for r in range(study.N_REPLICATES)
        }
        assert len(draws) == study.N_REPLICATES

    def test_a_cohort_whose_cells_no_longer_match_the_design_is_refused(self):
        # The stated counts were set for 17 slides per Camelyon cell; a cohort that grew or
        # shrank must not be subsampled to counts that no longer mean the stated fraction.
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=16, tiles_per_unit=2)
        with pytest.raises(ValueError, match="17"):
            study.draw_subsample(manifest, "pathorob-camelyon", 50, 0)

    def test_the_design_covers_three_cohorts_at_three_fractions_with_twenty_replicates(
        self,
    ):
        assert study.COHORTS == (
            "pathorob-camelyon",
            "pathorob-tcga-4x4",
            "pathorob-tolkach-esca",
        )
        assert study.FRACTIONS == (25, 50, 75)
        assert study.N_REPLICATES == 20


# --------------------------------------------------------------------------- replicates

TOY_MODELS = ("Alpha", "Beta", "DINOv2-B")
TOY_COHORT = "pathorob-camelyon"


def _toy_features(manifest: pd.DataFrame, model: str) -> np.ndarray:
    """Class and centre signal at a model-specific balance, plus per-slide jitter."""
    rng = np.random.default_rng(abs(hash(model)) % (2**32))
    bio = {"Alpha": 2.0, "Beta": 1.0, "DINOv2-B": 0.3}[model]
    site = {"Alpha": 0.5, "Beta": 1.2, "DINOv2-B": 0.3}[model]
    y = pd.factorize(manifest["label"], sort=True)[0]
    c = pd.factorize(manifest["medical_center"], sort=True)[0]
    g = pd.factorize(manifest["group_id"], sort=True)[0]
    slide = rng.normal(0.0, 0.4, size=(g.max() + 1, 6))[g]
    x = rng.normal(0.0, 1.0, size=(len(manifest), 6)) + slide
    x[:, 0] += bio * (2 * y - 1)
    x[:, 1] += site * (2 * c - 1)
    return x.astype(np.float32)


@pytest.fixture
def toy_production(bench_env):
    """A Camelyon-shaped production run (17 slides per cell) on a synthetic tileset."""
    manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=17, tiles_per_unit=3)
    bench_env.write_tileset(
        TOY_COHORT, manifest, {m: _toy_features(manifest, m) for m in TOY_MODELS}
    )
    bench_env.register(TOY_COHORT, tileset=TOY_COHORT, manifest=manifest, design="all", k_max=30)
    assert bench_env.run(TOY_COHORT, "median-k", "--k-grid", "sparse") == 0
    return manifest


def _text_metrics(mari: dict[str, str]) -> pd.DataFrame:
    """A ``metrics.csv`` as stored text: one row per model, with a path column."""
    return pd.DataFrame(
        {
            "model": list(mari),
            "mari": list(mari.values()),
            "embedding_path": [f"/somewhere/{m}.npy" for m in mari],
        }
    )


def _family_metrics(rows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """A ``metrics.csv`` as stored text with two column families: ``{model: (ri, croma)}``."""
    return pd.DataFrame(
        {
            "model": list(rows),
            "ri": [ri for ri, _ in rows.values()],
            "croma": [croma for _, croma in rows.values()],
            "embedding_path": [f"/somewhere/{m}.npy" for m in rows],
        }
    )


def _kernels(assignment: dict) -> dict[str, str]:
    return {model: assigned.kernel for model, assigned in assignment.items()}


TOLKACH = "pathorob-tolkach-esca"


class TestReplicates:
    def test_the_production_run_is_read_from_its_own_sidecar_and_metrics(self, toy_production):
        run = study.production_run(TOY_COHORT)
        assert run.roster == TOY_MODELS
        assert "--k-grid" in run.replay and "sparse" in run.replay
        assert "--tau" not in run.replay
        assert run.shared_k == int(run.metrics["k"].iloc[0])

    def test_the_full_fraction_reproduces_the_production_metrics_exactly(
        self, toy_production, tmp_path
    ):
        run = study.production_run(TOY_COHORT)
        full = study.draw_subsample(toy_production, TOY_COHORT, study.FULL, 0)
        out = tmp_path / "study" / "full"
        # "" runs in this process, with whatever GEMM kernel it already has.
        report = study.reproduce(
            run, full, out_dir=out, scratch=tmp_path / "scratch", candidates=("",)
        )
        assert report["identical"] is True
        assert set(report["variants"]) == {"reselected", "fixed-k"}
        assert report["kernel_by_model"] == {m: "" for m in TOY_MODELS}
        assert report["tolerance"] == 0.0
        assert report["max_deviation_by_model"] == {m: 0.0 for m in TOY_MODELS}
        assert study.reproduced(out.parent, TOY_COHORT) is False  # not a study tree
        assert (out / study.REPRODUCTION).exists()

    def test_a_single_differing_value_fails_the_reproduction_check(self, toy_production, tmp_path):
        run = study.production_run(TOY_COHORT)
        full = study.draw_subsample(toy_production, TOY_COHORT, study.FULL, 0)
        out = tmp_path / "study" / "full"
        study.reproduce(run, full, out_dir=out, scratch=tmp_path / "scratch", candidates=("",))
        metrics = out / "fixed-k" / "metrics.csv"
        frame = pd.read_csv(metrics, dtype=str, keep_default_na=False)
        # One ulp: the smallest change a float can make.
        frame.loc[1, "mari"] = repr(np.nextafter(float(frame.loc[1, "mari"]), 2.0))
        frame.to_csv(metrics, index=False)
        with pytest.raises(study.ReproductionError, match=r"fixed-k: \['mari'\]"):
            study.check_reproduction(run, out)

    def test_each_model_takes_the_kernel_its_published_row_was_computed_with(self):
        published = _text_metrics({"A": "0.25", "B": "0.5"})
        candidates = {
            "Cooperlake": {
                v: _text_metrics({"A": "0.25", "B": "0.5000000000000001"}) for v in study.VARIANTS
            },
            "SkylakeX": {
                v: _text_metrics({"A": "0.2500000000000001", "B": "0.5"}) for v in study.VARIANTS
            },
        }
        assert _kernels(study.assign_kernels(candidates, published)) == {
            "A": "Cooperlake",
            "B": "SkylakeX",
        }

    def test_a_model_both_kernels_reproduce_takes_the_first_candidate(self):
        published = _text_metrics({"A": "0.25"})
        candidates = {
            k: {v: _text_metrics({"A": "0.25"}) for v in study.VARIANTS}
            for k in ("SkylakeX", "Cooperlake")
        }
        assert study.assign_kernels(candidates, published) == {"A": ("SkylakeX", 0.0)}

    def test_a_model_only_one_variant_reproduces_under_a_kernel_is_not_assigned_it(
        self,
    ):
        published = _text_metrics({"A": "0.25", "B": "0.5"})
        exact = _text_metrics({"A": "0.25", "B": "0.5"})
        off = _text_metrics({"A": "0.25", "B": "0.5000000000000001"})
        candidates = {"Cooperlake": {study.RESELECTED: exact, study.FIXED_K: off}}
        with pytest.raises(study.ReproductionError, match=r"\['B'\]"):
            study.assign_kernels(candidates, published)

    def test_a_kernel_is_recovered_at_the_fixed_k_alone(self):
        # A single-kernel trial re-selects its shared k from that kernel's k* alone, which
        # can differ from the published mixed-kernel k: every re-selected row then differs,
        # while the fixed-k rows sit at the published operating point and still match.
        published = _text_metrics({"A": "0.25", "B": "0.5"})
        exact = _text_metrics({"A": "0.25", "B": "0.5"})
        elsewhere = _text_metrics({"A": "0.3", "B": "0.6"})
        candidates = {"Cooperlake": {study.RESELECTED: elsewhere, study.FIXED_K: exact}}
        assert _kernels(study.assign_kernels(candidates, published)) == {
            "A": "Cooperlake",
            "B": "Cooperlake",
        }

    def test_only_tolkach_esca_is_checked_within_a_tolerance(self):
        assert set(study.REPRODUCTION_TOLERANCE) == set(study.COHORTS)
        assert {c for c, t in study.REPRODUCTION_TOLERANCE.items() if t > 0} == {TOLKACH}

    def test_a_row_split_across_kernels_by_column_family_is_assigned_within_the_tolerance(
        self,
    ):
        # RI from one kernel and CRoMa from the other, as the published Tolkach-ESCA run
        # assembled Mettle's row: no kernel reproduces it exactly. The kernel that deviates
        # least wins, though it is listed second.
        published = _family_metrics({"M": ("0.5", "0.25")})
        candidates = {
            "Cooperlake": {v: _family_metrics({"M": ("0.50002", "0.25")}) for v in study.VARIANTS},
            "SkylakeX": {v: _family_metrics({"M": ("0.5", "0.25000003")}) for v in study.VARIANTS},
        }
        tolerance = study.reproduction_tolerance(TOLKACH)
        assignment = study.assign_kernels(candidates, published, tolerance=tolerance)
        assert assignment == {"M": ("SkylakeX", abs(0.25000003 - 0.25))}
        assert assignment["M"].max_deviation == abs(0.25000003 - 0.25)
        # Checked exactly, as on Camelyon and TCGA-4x4, the same row reproduces under neither.
        with pytest.raises(study.ReproductionError, match=r"\['M'\]"):
            study.assign_kernels(candidates, published)

    def test_a_row_no_kernel_reproduces_within_the_tolerance_still_raises(self):
        published = _family_metrics({m: ("0.5", "0.25") for m in ("M", "N", "P")})
        candidates = {}
        for kernel, ri in (("Cooperlake", "0.501"), ("SkylakeX", "0.4998")):
            # N is off by more than the tolerance under both kernels; P's numbers match, but
            # a value that is not a number must still match exactly.
            trial = _family_metrics({"M": ("0.5", "0.25"), "N": (ri, "0.25"), "P": ("0.5", "0.25")})
            trial.loc[2, "embedding_path"] = "/somewhere/other.npy"
            candidates[kernel] = {v: trial for v in study.VARIANTS}
        with pytest.raises(study.ReproductionError, match=r"\['N', 'P'\] .* within 0\.0001"):
            study.assign_kernels(
                candidates, published, tolerance=study.reproduction_tolerance(TOLKACH)
            )

    def test_a_reproduction_passes_only_within_its_cohort_s_tolerance(self):
        within = {"identical": False, "max_deviation": 1.1e-7}
        # An exact record passes anywhere, including one written before tolerances existed.
        assert study.reproduction_passed({"identical": True}, "pathorob-tcga-4x4")
        assert study.reproduction_passed(within, TOLKACH)
        assert not study.reproduction_passed(within, "pathorob-tcga-4x4")
        assert not study.reproduction_passed({"identical": False, "max_deviation": 2e-4}, TOLKACH)
        assert not study.reproduction_passed({}, TOLKACH)

    def test_the_recorded_max_deviation_is_the_computed_one(
        self, toy_production, tmp_path, monkeypatch
    ):
        run = study.production_run(TOY_COHORT)
        full = study.draw_subsample(toy_production, TOY_COHORT, study.FULL, 0)
        out = tmp_path / "study" / "full"
        study.reproduce(run, full, out_dir=out, scratch=tmp_path / "scratch", candidates=("",))
        published = study._as_written(run.results_dir / "metrics.csv")
        # One model's MaRI is off under one protocol and its CRoMa under the other.
        record = {}
        for variant, column, shift in (
            (study.FIXED_K, "mari", 3e-5),
            (study.RESELECTED, "croma", -7e-6),
        ):
            path = out / variant / "metrics.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.loc[1, column] = repr(float(frame.loc[1, column]) + shift)
            frame.to_csv(path, index=False)
            record[column] = abs(float(frame.loc[1, column]) - float(published.loc[1, column]))
        tolerance = study.reproduction_tolerance(TOLKACH)
        monkeypatch.setitem(study.REPRODUCTION_TOLERANCE, TOY_COHORT, tolerance)

        report = study.check_reproduction(run, out)
        computed = max(record.values())
        assert report["max_deviation_by_model"] == {
            m: computed if m == TOY_MODELS[1] else 0.0 for m in TOY_MODELS
        }
        assert report["max_deviation"] == computed
        assert (report["identical"], report["tolerance"]) == (False, tolerance)
        assert json.loads((out / study.REPRODUCTION).read_text()) == report
        assert study.reproduction_passed(report, TOY_COHORT)
        # Tightening the tolerance invalidates a record it no longer covers.
        monkeypatch.setitem(study.REPRODUCTION_TOLERANCE, TOY_COHORT, 0.0)
        assert not study.reproduction_passed(report, TOY_COHORT)
        with pytest.raises(study.ReproductionError, match=r"fixed-k: \['mari'\]"):
            study.check_reproduction(run, out)

    def test_a_replicate_records_its_shared_k_every_k_star_and_ri_support(
        self, toy_production, tmp_path
    ):
        run = study.production_run(TOY_COHORT)
        drawn = study.draw_subsample(toy_production, TOY_COHORT, 25, 0)
        out = tmp_path / "study" / "r00"
        scores = study.run_replicate(run, drawn, out_dir=out, scratch=tmp_path / "scratch")

        assert list(scores["model"]) == list(TOY_MODELS)
        meta = study.read_replicate(out)
        assert meta["n_rows"] == len(drawn)
        # The re-selected shared k is the lower median of this replicate's own k*.
        k_star = dict(zip(scores["model"], scores["k_star"]))
        assert meta["shared_k"] == sorted(k_star.values())[1]
        assert set(scores["k"]) == {meta["shared_k"]}
        # The secondary protocol holds the full-cohort operating point fixed.
        assert meta["fixed_k"] == run.shared_k
        assert set(scores["k_fixed_k"]) == {run.shared_k}
        for column in ("support", "support_fixed_k"):
            assert scores[column].between(0.0, 1.0).all()
        # One draw: CRoMa is k-free, so both runs scored the same rows identically.
        for column in ("croma", "croma_f0", "croma_ltm_alpha"):
            assert scores[column].notna().all()

    def test_the_scratch_tree_is_removed_and_the_driver_state_restored(
        self, toy_production, tmp_path
    ):
        import benchmark as bm
        import benchmarks
        import layout

        run = study.production_run(TOY_COHORT)
        before = (layout.OUTPUT_ROOT, benchmarks.get(TOY_COHORT), bm._shared_operating_k)
        drawn = study.draw_subsample(toy_production, TOY_COHORT, 50, 1)
        scratch = tmp_path / "scratch"
        study.run_replicate(run, drawn, out_dir=tmp_path / "r01", scratch=scratch)
        after = (layout.OUTPUT_ROOT, benchmarks.get(TOY_COHORT), bm._shared_operating_k)
        assert after == before
        assert not scratch.exists()

    def test_a_k_grid_reaching_past_the_usable_neighbours_is_refused(self, bench_env, tmp_path):
        # 2 slides x 3 tiles x 4 cells = 24 rows; a slide's own 3 tiles are excluded, so
        # 21 neighbours are usable -- and the sparse grid at k_max=30 reaches k=21.
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=17, tiles_per_unit=3)
        bench_env.write_tileset(
            TOY_COHORT, manifest, {m: _toy_features(manifest, m) for m in TOY_MODELS}
        )
        bench_env.register(
            TOY_COHORT, tileset=TOY_COHORT, manifest=manifest, design="all", k_max=30
        )
        assert bench_env.run(TOY_COHORT, "median-k", "--k-grid", "sparse") == 0
        run = study.production_run(TOY_COHORT)
        tiny = manifest[manifest["group_id"].str.endswith(("_0", "_1"))]
        with pytest.raises(ValueError, match="usable neighbours"):
            study.run_replicate(run, tiny, out_dir=tmp_path / "r", scratch=tmp_path / "scratch")


# --------------------------------------------------------------------------- measures

if str(ROOT / "scripts" / "repro") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "repro"))

import _subsampling as measures  # noqa: E402

CONTROL = "DINOv2-B"
ENCODERS = ("A", "B", "C", "D")
METRIC_COLUMNS = (
    "ri",
    "ri_fixed_k",
    "mari",
    "mari_fixed_k",
    "croma",
    "croma_ltm_alpha",
)


def _record(out_dir: Path, scores: dict, *, shared_k: int, fixed_k: int = 11) -> None:
    """Write one replicate's record as the sweep does, from per-column {model: value}."""
    models = (*ENCODERS, CONTROL)
    frame = pd.DataFrame({"model": models})
    for column in ("support_fixed_k", "croma_f0", "tau", "tau_fixed_k"):
        frame[column] = 0.5
    for column, values in scores.items():
        frame[column] = [values[m] for m in models]
    frame["k"] = shared_k
    frame["k_fixed_k"] = fixed_k
    meta = {
        "benchmark": "pathorob-camelyon",
        "n_rows": 100,
        "n_units": 10,
        "shared_k": shared_k,
        "fixed_k": fixed_k,
        "k_grid_max": 91,
        "min_usable_neighbours": 99,
        "roster": list(models),
    }
    study.record_replicate(out_dir, frame, meta)


def _scores(values, control):
    return dict(zip((*ENCODERS, CONTROL), (*values, control)))


FULL_SCORES = _scores((4.0, 3.0, 2.0, 1.0), 0.0)


@pytest.fixture
def small_study(tmp_path):
    """One cohort, one fraction, three replicates, with hand-worked summaries.

    The control scores 10 in every replicate and 0 on the full cohort: ranked with the
    encoders it would move every number below, so each assertion also checks it was left out.
    """
    root = tmp_path / "subsampling-stability"
    design = study.design(cohorts=("pathorob-camelyon",), fractions=(25,), n_replicates=3)
    study.write_design(root, design)
    cohort = "pathorob-camelyon"
    full_dir = study.replicate_dir(root, cohort, study.FULL, 0)
    _record(
        full_dir,
        {
            **{c: FULL_SCORES for c in METRIC_COLUMNS},
            "support": _scores((0.5, 0.6, 0.7, 0.8), 0.0),
            "k_star": _scores((11, 11, 21, 5), 1),
        },
        shared_k=11,
    )
    (full_dir / study.REPRODUCTION).write_text('{"identical": true}\n')

    croma = [
        _scores((4.1, 3.1, 2.1, 1.1), 10.0),  # same order, every score +0.1
        _scores((3.8, 2.8, 1.8, 0.8), 10.0),  # same order, every score -0.2
        _scores((4.5, 3.5, 1.0, 2.0), 10.0),  # C and D swap
    ]
    ri = [
        _scores((3.0, 4.0, 2.0, 1.0), 10.0),  # A and B swap
        _scores((1.0, 2.0, 3.0, 4.0), 10.0),  # reversed
        _scores((4.0, 3.0, 2.0, 1.0), 10.0),
    ]
    k_star = [
        _scores((11, 11, 21, 5), 3),
        _scores((21, 11, 21, 5), 3),  # A moved
        _scores((21, 5, 21, 5), 3),  # A and B moved
    ]
    support = [_scores((v,) * 4, 0.0) for v in (0.4, 0.5, 0.6)]
    for r, shared_k in enumerate((11, 21, 9)):
        _record(
            study.replicate_dir(root, cohort, 25, r),
            {
                **{c: _scores((4.0, 3.0, 2.0, 1.0), 10.0) for c in METRIC_COLUMNS},
                "croma": croma[r],
                "ri": ri[r],
                "k_star": k_star[r],
                "support": support[r],
            },
            shared_k=shared_k,
        )
    return root, design


def _row(frame: pd.DataFrame, **where) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in where.items():
        mask &= frame[key].to_numpy() == value
    assert mask.sum() == 1, where
    return frame[mask].iloc[0]


class TestMeasures:
    def test_kendall_tau_k_agreement_is_summarised_by_median_and_iqr(self, small_study):
        root, design = small_study
        ranking = measures.ranking_summary(measures.load(root, expected=design))
        # croma: tau_K = 1, 1, 2/3 (one discordant pair of six).
        row = _row(ranking, cohort="pathorob-camelyon", fraction=25, metric="croma")
        assert row["tau_k_median"] == pytest.approx(1.0)
        assert row["tau_k_q25"] == pytest.approx(5 / 6)
        assert row["tau_k_q75"] == pytest.approx(1.0)
        assert row["n_replicates"] == 3 and row["n_encoders"] == 4
        # ri: tau_K = 2/3, -1, 1.
        row = _row(ranking, cohort="pathorob-camelyon", fraction=25, metric="ri")
        assert row["tau_k_median"] == pytest.approx(2 / 3)

    def test_rank_interval_width_is_the_median_over_encoders_of_the_95pct_range(self, small_study):
        root, design = small_study
        ranking = measures.ranking_summary(measures.load(root, expected=design))
        row = _row(ranking, cohort="pathorob-camelyon", fraction=25, metric="croma")
        # A and B keep their ranks; C's ranks 3,3,4 and D's 4,4,3 span 0.95 each.
        assert row["rank_width_median"] == pytest.approx(0.475)

    def test_drift_is_the_signed_median_of_subsample_minus_full(self, small_study):
        root, design = small_study
        ranking = measures.ranking_summary(measures.load(root, expected=design))
        row = _row(ranking, cohort="pathorob-camelyon", fraction=25, metric="croma")
        assert row["drift_median"] == pytest.approx(0.1)

    def test_the_full_cohort_sits_at_perfect_agreement(self, small_study):
        root, design = small_study
        ranking = measures.ranking_summary(measures.load(root, expected=design))
        for metric in measures.METRICS:
            row = _row(ranking, cohort="pathorob-camelyon", fraction=100, metric=metric)
            assert row["tau_k_median"] == 1.0
            assert row["rank_width_median"] == 0.0

    def test_the_paired_comparison_differences_tau_k_within_each_replicate(self, small_study):
        root, design = small_study
        paired = measures.paired_summary(measures.load(root, expected=design))
        # croma - ri per replicate: 1/3, 2, -1/3.
        row = _row(paired, cohort="pathorob-camelyon", fraction=25, versus="ri")
        assert row["tau_k_diff_median"] == pytest.approx(1 / 3)
        assert row["tau_k_diff_lo"] == pytest.approx(-0.3)
        assert row["tau_k_diff_hi"] == pytest.approx(1 / 3 + 0.95 * 5 / 3)
        assert row["n_croma_higher"] == 2 and row["n_replicates"] == 3
        # croma - mari: 0, 0, -1/3 -- a tie is not "higher".
        row = _row(paired, cohort="pathorob-camelyon", fraction=25, versus="mari")
        assert row["tau_k_diff_median"] == pytest.approx(0.0)
        assert row["n_croma_higher"] == 0

    def test_k_and_support_are_summarised_against_the_full_cohort(self, small_study):
        root, design = small_study
        k = measures.k_support_summary(measures.load(root, expected=design))
        row = _row(k, cohort="pathorob-camelyon", fraction=25)
        assert (row["shared_k_min"], row["shared_k_max"]) == (9, 21)
        assert row["full_shared_k"] == 11
        assert row["k_star_changed_share"] == pytest.approx(3 / 12)
        assert row["support_median"] == pytest.approx(0.5)
        assert row["full_support_median"] == pytest.approx(0.65)

    def test_scores_one_ulp_apart_load_exactly_and_keep_their_order(self, tmp_path):
        # The sweep writes shortest round-trip floats. pandas' default parser can read two
        # adjacent doubles back as one value, which would turn this swap into a tie.
        lo, hi = 0.1, float(np.nextafter(0.1, 1.0))
        cohort = "pathorob-camelyon"
        root = tmp_path / "subsampling-stability"
        design = study.design(cohorts=(cohort,), fractions=(25,), n_replicates=1)
        study.write_design(root, design)
        common = {"support": _scores((0.5,) * 4, 0.0), "k_star": _scores((11,) * 4, 1)}
        full_dir = study.replicate_dir(root, cohort, study.FULL, 0)
        full_scores = _scores((0.3, lo, hi, 0.001), 0.0)
        _record(full_dir, {**{c: full_scores for c in METRIC_COLUMNS}, **common}, shared_k=11)
        (full_dir / study.REPRODUCTION).write_text('{"identical": true}\n')
        swapped = _scores((0.3, hi, lo, 0.001), 0.0)  # B and C swap
        _record(
            study.replicate_dir(root, cohort, 25, 0),
            {**{c: swapped for c in METRIC_COLUMNS}, **common},
            shared_k=11,
        )

        data = measures.load(root, expected=design)
        replicate = data.scores[data.scores["fraction"] == 25].set_index("model")
        for metric in measures.METRICS:
            assert (replicate.loc["B", metric], replicate.loc["C", metric]) == (hi, lo)
        tau = measures.kendall_tau_k(data)
        for metric in measures.METRICS:
            # One discordant pair of six.
            row = _row(tau, cohort=cohort, fraction=25, metric=metric)
            assert row["tau_k"] == pytest.approx(2 / 3)

    def test_a_missing_replicate_fails_the_load(self, small_study):
        root, design = small_study
        (study.replicate_dir(root, "pathorob-camelyon", 25, 1) / study.SCORES).unlink()
        with pytest.raises(measures.IncompleteStudyError, match="r01"):
            measures.load(root, expected=design)

    def test_a_full_cohort_without_a_passed_reproduction_fails_the_load(self, small_study):
        root, design = small_study
        full = study.replicate_dir(root, "pathorob-camelyon", study.FULL, 0)
        (full / study.REPRODUCTION).unlink()
        with pytest.raises(measures.IncompleteStudyError, match="reproduction"):
            measures.load(root, expected=design)

    def test_a_full_cohort_reproduced_only_within_a_tolerance_it_lacks_fails_the_load(
        self, small_study
    ):
        root, design = small_study
        full = study.replicate_dir(root, "pathorob-camelyon", study.FULL, 0)
        within = {"identical": False, "tolerance": 1e-4, "max_deviation": 1e-7}
        (full / study.REPRODUCTION).write_text(json.dumps(within) + "\n")
        with pytest.raises(measures.IncompleteStudyError, match="reproduction"):
            measures.load(root, expected=design)

    def test_an_exact_study_says_so_in_one_sentence(self, small_study):
        root, design = small_study
        data = measures.load(root, expected=design)
        assert floats._reproduction_clause(data) == (
            "The full-cohort run reproduces every published score exactly."
        )

    def test_a_study_drawn_under_another_design_fails_the_load(self, small_study):
        root, _ = small_study
        with pytest.raises(measures.IncompleteStudyError, match="design"):
            measures.load(root)  # the real design: three cohorts, 20 replicates


# --------------------------------------------------------------------------- rendering

if str(ROOT / "scripts" / "repro" / "figures") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "repro" / "figures"))

import generate_subsampling_stability_float as floats  # noqa: E402
import subsampling_stability_figure as figure  # noqa: E402

PANEL = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta")
GRID = (1, 3, 5, 7, 9, 11, 21, 31, 41, 51, 61, 71, 81, 91)
FULL_K = {"pathorob-camelyon": 11, "pathorob-tcga-4x4": 81, "pathorob-tolkach-esca": 71}
#: How each full-cohort run reproduced production: exactly, but Tolkach-ESCA within tolerance.
REPRODUCTION_RECORD = {
    "pathorob-camelyon": {"identical": True},
    "pathorob-tcga-4x4": {"identical": True},
    "pathorob-tolkach-esca": {
        "identical": False,
        "tolerance": 1e-4,
        "max_deviation": 1.1091463059109685e-07,
    },
}


def _write_real_design_study(root: Path, seed: int) -> None:
    """A complete study under the real design, with seeded synthetic scores."""
    rng = np.random.default_rng(seed)
    study.write_design(root, study.design())
    models = (*PANEL, CONTROL)
    for cohort in study.COHORTS:
        base = {c: rng.normal(0.0, 1.0, len(models)) for c in METRIC_COLUMNS}
        full_k_star = rng.choice(GRID, len(models))
        tasks = [(study.FULL, 0)] + [
            (f, r) for f in study.FRACTIONS for r in range(study.N_REPLICATES)
        ]
        for fraction, replicate in tasks:
            noise = 0.0 if fraction == study.FULL else (100 - fraction) / 100
            frame = pd.DataFrame({"model": models})
            for column in METRIC_COLUMNS:
                frame[column] = base[column] + rng.normal(0.0, noise, len(models))
            frame["croma_f0"] = 0.5
            k_star = (
                full_k_star
                if fraction == study.FULL
                else np.where(
                    rng.random(len(models)) < 0.3, rng.choice(GRID, len(models)), full_k_star
                )
            )
            frame["k_star"] = k_star
            shared = int(np.sort(k_star)[(len(models) - 1) // 2])
            if fraction == study.FULL:
                shared = FULL_K[cohort]
            frame["k"], frame["k_fixed_k"] = shared, FULL_K[cohort]
            frame["support"] = rng.uniform(0.2, 0.9, len(models))
            frame["support_fixed_k"] = frame["support"]
            frame["tau"] = frame["tau_fixed_k"] = 0.1
            meta = {
                "benchmark": cohort,
                "n_rows": 100,
                "n_units": 10,
                "shared_k": shared,
                "fixed_k": FULL_K[cohort],
                "k_grid_max": 91,
                "min_usable_neighbours": 1000,
                "roster": list(models),
                "kernel_by_model": {m: "" for m in models},
            }
            where = study.replicate_dir(root, cohort, fraction, replicate)
            study.record_replicate(where, frame, meta)
            if fraction == study.FULL:
                record = json.dumps(REPRODUCTION_RECORD[cohort])
                (where / study.REPRODUCTION).write_text(record + "\n")


@pytest.fixture(scope="module")
def real_design_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("subsampling") / "subsampling-stability"
    _write_real_design_study(root, seed=0)
    return measures.load(root)


class TestRendering:
    def test_the_table_has_a_row_per_cohort_and_metric_and_per_cohort_and_fraction(
        self, real_design_study
    ):
        tex = floats.build_table(real_design_study)
        assert r"\label{tab:subsampling-stability}" in tex
        assert tex.count(r"\label{") == 1
        assert tex.startswith("% AUTO-GENERATED")
        for name in ("Camelyon", "TCGA-4x4", "Tolkach-ESCA"):
            assert name in tex
        # (a) one row per cohort x metric; (b) one row per cohort x fraction.
        body = [line for line in tex.splitlines() if line.rstrip().endswith(r"\\")]
        header_rows = 3
        assert len(body) == 3 * len(measures.METRICS) + 3 * 3 + header_rows

    def test_the_table_caption_states_how_each_full_cohort_run_reproduced(self, real_design_study):
        caption = _caption(floats.build_table(real_design_study))
        assert (
            "The full-cohort run reproduces the published scores exactly on Camelyon and "
            "TCGA-4x4, and on Tolkach-ESCA only within an absolute tolerance of $10^{-4}$ "
            r"(largest deviation $1.1\times10^{-7}$), because its published run mixed two "
            "BLAS kernels within an encoder's scores."
        ) in caption

    def test_the_figure_float_includes_the_rendered_figure(self, real_design_study):
        tex = floats.build_figure(real_design_study)
        assert r"\label{fig:subsampling-stability}" in tex
        assert r"\includegraphics" in tex and "subsampling_stability.pdf" in tex
        assert "[!htbp]" in tex

    def test_rebuilding_is_byte_for_byte(self, real_design_study, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        for out in (first, second):
            floats.write(real_design_study, out)
        for name in (floats.TABLE_NAME, floats.FIGURE_NAME):
            assert (first / name).read_bytes() == (second / name).read_bytes()

    def test_the_captions_narrate_no_result(self, real_design_study, tmp_path):
        """Captions change with the design, never with the scores."""
        other_root = tmp_path / "subsampling-stability"
        _write_real_design_study(other_root, seed=1)
        other = measures.load(other_root)
        for build in (floats.build_table, floats.build_figure):
            assert _caption(build(real_design_study)) == _caption(build(other))
        assert floats.build_table(real_design_study) != floats.build_table(other)

    def test_each_caption_is_within_the_350_word_legend_limit(self, real_design_study):
        for build in (floats.build_table, floats.build_figure):
            words = re.findall(
                r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", _caption(build(real_design_study))
            )
            assert len(words) <= 350

    def test_the_figure_renders_beside_the_study(self, real_design_study, tmp_path):
        out = tmp_path / "plots" / "subsampling_stability.png"
        figure.render(real_design_study, out)
        assert (tmp_path / "plots" / "png" / "subsampling_stability.png").stat().st_size
        assert (tmp_path / "plots" / "pdf" / "subsampling_stability.pdf").stat().st_size

    def test_the_macros_cover_every_cohort_metric_and_comparison(self, real_design_study):
        lines = measures.macros(real_design_study)
        parsed = [re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{.+\}", line) for line in lines]
        assert all(parsed), [line for line, m in zip(lines, parsed) if not m]
        names = [m.group(1) for m in parsed]
        assert len(names) == len(set(names))
        prefixes = ("Camelyon", "TcgaFourByFour", "Tolkach")
        for prefix in prefixes:
            for metric in ("Ri", "RiFixedK", "Mari", "MariFixedK", "Croma", "Ltm"):
                assert f"{prefix}SubsampleHalfKendall{metric}" in names
            for against in ("Ri", "Mari"):
                for stat in ("", "Range", "Higher"):
                    assert f"{prefix}SubsampleHalfCromaMinus{against}{stat}" in names
            assert f"{prefix}SubsampleQuarterSharedKRange" in names
            for stat in ("Tolerance", "MaxDeviation"):
                assert f"{prefix}SubsampleReproduction{stat}" in names
        assert not any(c.isdigit() for name in names for c in name)

    def test_a_macro_carries_the_number_it_names(self, real_design_study):
        ranking = measures.ranking_summary(real_design_study)
        row = _row(ranking, cohort="pathorob-tolkach-esca", fraction=50, metric="mari")
        lines = measures.macros(real_design_study)
        (line,) = [l for l in lines if r"{\TolkachSubsampleHalfKendallMari}" in l]
        assert line.endswith("{" + f"${row['tau_k_median']:.2f}$" + "}")
        assert r"\newcommand{\TolkachSubsampleReproductionTolerance}{$10^{-4}$}" in lines
        assert (
            r"\newcommand{\TolkachSubsampleReproductionMaxDeviation}{$1.1\times10^{-7}$}" in lines
        )
        assert r"\newcommand{\CamelyonSubsampleReproductionMaxDeviation}{$0$}" in lines


class TestPaperValues:
    def test_the_values_file_carries_the_macros_only_once_the_study_is_complete(
        self, tmp_path, monkeypatch
    ):
        import generate_paper_values as values

        # The frontier guard reads the live benchmark runs, which a checkout may not have.
        monkeypatch.setattr(values, "_load_rank_frontier_change_macros", lambda: [])
        root = tmp_path / "output" / "studies" / "subsampling-stability"
        _write_real_design_study(root, seed=0)
        cited = r"\newcommand{\TolkachSubsampleHalfKendallMari}"
        assert cited not in values.build([], tmp_path, "auto")
        assert study.mark_complete(root, study.design())
        tex = values.build([], tmp_path, "auto")
        assert cited in tex
        assert tex.count(r"\newcommand{\SubsampleNReplicates}{20}") == 1


def _caption(tex: str) -> str:
    start = tex.index(r"\caption{")
    return tex[start : tex.index(r"\label{", start)]


# --------------------------------------------------------------------------- the sweep

TWO_COHORTS = ("pathorob-camelyon", "pathorob-tolkach-esca")


def _plan_design():
    return study.design(cohorts=TWO_COHORTS, fractions=(25, 75), n_replicates=2)


def _finish(root, cohort, fraction, replicate, *, reproduced=True):
    where = study.replicate_dir(root, cohort, fraction, replicate)
    _record(where, {c: FULL_SCORES for c in METRIC_COLUMNS}, shared_k=11)
    if fraction == study.FULL and reproduced:
        (where / study.REPRODUCTION).write_text('{"identical": true}\n')


class TestSweepPlan:
    def test_a_fresh_study_first_runs_every_full_cohort_and_nothing_else(self, tmp_path):
        full, ready = study.plan(tmp_path, _plan_design())
        assert full == [(c, study.FULL, 0) for c in TWO_COHORTS]
        assert ready == []

    def test_a_cohort_s_replicates_wait_for_its_reproduction_to_pass(self, tmp_path):
        _finish(tmp_path, "pathorob-camelyon", study.FULL, 0)
        # Tolkach-ESCA's full run finished but its reproduction never passed.
        _finish(tmp_path, "pathorob-tolkach-esca", study.FULL, 0, reproduced=False)
        full, ready = study.plan(tmp_path, _plan_design())
        assert full == [("pathorob-tolkach-esca", study.FULL, 0)]
        assert sorted(ready) == [("pathorob-camelyon", f, r) for f in (25, 75) for r in (0, 1)]

    def test_a_resumed_sweep_skips_every_finished_replicate(self, tmp_path):
        for cohort in TWO_COHORTS:
            _finish(tmp_path, cohort, study.FULL, 0)
        _finish(tmp_path, "pathorob-camelyon", 25, 1)
        _finish(tmp_path, "pathorob-tolkach-esca", 75, 0)
        # A replicate that died before its completion mark is run again.
        partial = study.replicate_dir(tmp_path, "pathorob-camelyon", 75, 0)
        (partial / study.FIXED_K).mkdir(parents=True)
        full, ready = study.plan(tmp_path, _plan_design())
        assert full == []
        assert len(ready) == 2 * 2 * 2 - 2
        assert ("pathorob-camelyon", 25, 1) not in ready
        assert ("pathorob-tolkach-esca", 75, 0) not in ready
        assert ("pathorob-camelyon", 75, 0) in ready

    def test_status_counts_finished_against_expected_replicates(self, tmp_path):
        _finish(tmp_path, "pathorob-camelyon", study.FULL, 0)
        _finish(tmp_path, "pathorob-camelyon", 25, 0)
        status = study.status(tmp_path, _plan_design())
        row = status[(status["cohort"] == "pathorob-camelyon") & (status["fraction"] == 25)]
        assert (int(row["finished"].iloc[0]), int(row["expected"].iloc[0])) == (1, 2)
        assert int(status["finished"].sum()) == 2
        assert int(status["expected"].sum()) == 2 + 2 * 2 * 2

    def test_only_a_finished_study_is_marked_complete(self, tmp_path):
        record = _plan_design()
        tasks = [(c, study.FULL, 0) for c in TWO_COHORTS] + [
            (c, f, r) for c in TWO_COHORTS for f in (25, 75) for r in (0, 1)
        ]
        for task in tasks[:-1]:
            _finish(tmp_path, *task)
        assert study.mark_complete(tmp_path, record) is False
        assert not (tmp_path / study.COMPLETE).exists()
        _finish(tmp_path, *tasks[-1])
        assert study.mark_complete(tmp_path, record) is True
        assert (tmp_path / study.COMPLETE).exists()


class TestCommandLine:
    """Workers run from the repository, so a relative path must mean the caller's."""

    @pytest.fixture
    def received(self, tmp_path, monkeypatch):
        # Launched outside the repository, with the study root reached through a symlink
        # (as output/ is): the path is made absolute, never resolved through the link.
        (tmp_path / "shared").mkdir()
        (tmp_path / "output").symlink_to(tmp_path / "shared", target_is_directory=True)
        monkeypatch.chdir(tmp_path)
        calls: dict = {}

        def sweep(root, record, *, workers, scratch_root):
            calls.update(root=root, scratch_root=scratch_root)
            return 0

        def run_task(root, task, scratch_root):
            calls.update(root=root, scratch_root=scratch_root)

        monkeypatch.setattr(study, "sweep", sweep)
        monkeypatch.setattr(study, "run_task", run_task)
        return calls

    @pytest.mark.parametrize(
        "command",
        [
            ["sweep"],
            ["replicate", "--cohort", "pathorob-camelyon", "--fraction", "25", "--replicate", "0"],
        ],
    )
    def test_relative_root_and_scratch_are_taken_from_the_caller_s_directory(
        self, received, command
    ):
        argv = [*command, "--root", "output/study", "--scratch-root", "scratch"]
        assert study.main(argv) == 0
        assert received["root"] == Path.cwd() / "output" / "study"
        assert received["scratch_root"] == Path.cwd() / "scratch"
