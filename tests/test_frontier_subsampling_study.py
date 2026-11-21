"""Frontier membership under fixed and recomputed margins, on the subsampling study's draws.

Both arms score the anchors one subsampling replicate retained: the fixed arm summarises
their full-cohort margins, the recomputed arm is the replicate's own ``scores.csv``. These
tests pin what a reader has to trust before comparing the two: that the fixed arm reads
exactly the rows of the retained units, that those are the rows the replicate scored, and
that the frontier is the one ``bootstrap_uncertainty`` and the Pareto figures take.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts" / "studies", ROOT / "scripts" / "bench"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import bootstrap_uncertainty as bootstrap  # noqa: E402
import frontier_subsampling as study  # noqa: E402
import subsampling_stability as subsampling  # noqa: E402


def _cohort_manifest(
    *, classes: int, centres: int, units_per_cell: int, shared_cases: bool, tiles: int = 2
) -> pd.DataFrame:
    """Every class x centre cell holds ``units_per_cell`` blocks of ``tiles`` rows.

    With ``shared_cases`` one case contributes a block to every class, as on Tolkach-ESCA.
    """
    rows = []
    for c in range(centres):
        for y in range(classes):
            for u in range(units_per_cell):
                case = f"case{c}_{u}" if shared_cases else f"slide{c}_{y}_{u}"
                for t in range(tiles):
                    rows.append(
                        {
                            "sample_id": f"{case}__y{y}__t{t}",
                            "label": f"class{y}",
                            "medical_center": f"centre{c}",
                            "group_id": case,
                        }
                    )
    return pd.DataFrame(rows)


#: A Tolkach-shaped cohort: one centre, two classes, and case ``a`` contributing a block to
#: both classes. Rows 0-1 are a's class-x block, 2 is b's, 3-4 are a's class-y block, 5 is c's.
SHARED_CASE_MANIFEST = pd.DataFrame(
    {
        "sample_id": ["a_x0", "a_x1", "b_x0", "a_y0", "a_y1", "c_y0"],
        "label": ["x", "x", "x", "y", "y", "y"],
        "medical_center": ["k"] * 6,
        "group_id": ["a", "a", "b", "a", "a", "c"],
    }
)


class TestFixedArmRows:
    def test_the_fixed_arm_reads_exactly_the_rows_of_the_retained_units(self):
        # The replicate kept a's block in class x and c's block in class y. a's block in
        # class y shares its case but is another unit, so its rows (3, 4) stay out.
        units = pd.DataFrame(
            {"label": ["x", "y"], "medical_center": ["k", "k"], "group_id": ["a", "c"]}
        )
        rows = study.retained_rows(SHARED_CASE_MANIFEST, units)
        assert rows.tolist() == [0, 1, 5]

    def test_a_retained_unit_with_no_full_cohort_margin_is_refused(self):
        # b has no block in class y: a units.csv naming one was not drawn from this cohort,
        # and silently scoring the other units would compare the arms on different anchors.
        units = pd.DataFrame(
            {"label": ["x", "y"], "medical_center": ["k", "k"], "group_id": ["a", "b"]}
        )
        with pytest.raises(ValueError, match="'y', 'k', 'b'"):
            study.retained_rows(SHARED_CASE_MANIFEST, units)

    @pytest.mark.parametrize(
        "cohort, shared_cases",
        [("pathorob-camelyon", False), ("pathorob-tolkach-esca", True)],
    )
    def test_both_arms_evaluate_the_anchors_the_replicate_scored(self, cohort, shared_cases):
        # The recomputed arm is the driver run on the drawn manifest, so its anchors are the
        # drawn rows; units.csv is that draw's units, written as the sweep writes it.
        full, _ = subsampling.UNITS_PER_CELL[cohort]
        manifest = _cohort_manifest(
            classes=3, centres=2, units_per_cell=full, shared_cases=shared_cases
        )
        for fraction in subsampling.FRACTIONS:
            for replicate in range(3):
                drawn = subsampling.draw_subsample(manifest, cohort, fraction, replicate)
                units = drawn.loc[:, list(subsampling.UNIT)].drop_duplicates()
                rows = study.retained_rows(manifest, units)
                assert manifest["sample_id"].iloc[rows].tolist() == drawn["sample_id"].tolist()


class TestFixedArmSummaries:
    def test_the_fixed_arm_summarises_the_retained_rows_full_cohort_margins(self):
        # Rows 0-4 are retained. LTM_10 of five values is the lowest: the 10th percentile
        # sits 0.4 of the way from the first sorted value to the second. Row 5 is left out;
        # kept, it would move A's median to 3.5 and B's tail to -50.
        values = {
            "A": np.array([5.0, 1.0, 2.0, 3.0, 4.0, 100.0]),
            "B": np.array([-1.0, 0.0, 4.0, 5.0, 6.0, -50.0]),
        }
        summary = study.fixed_arm(values, np.arange(5))
        assert summary.loc["A"].to_dict() == {"median": 3.0, "ltm": 1.0}
        assert summary.loc["B"].to_dict() == {"median": 4.0, "ltm": -1.0}
        assert summary.index.tolist() == ["A", "B"]


def _frequencies(models, sets) -> dict[str, float]:
    """The frequencies the study writes, from one arm's sets (the other arm is the same)."""
    table = study.paired_membership(models, sets, sets).set_index("model")
    return table["freq_fixed"].to_dict()


class TestFrontier:
    def test_each_replicate_frontier_is_the_set_of_undominated_encoders(self):
        models = ["A", "B", "C"]
        # replicate 0: A (1, 0), B (0, 1), C (0, 0)   -> {A, B}, staircase B then A
        # replicate 1: A (1, 1), B (0, 0), C (0, 0.5) -> {A}
        medians = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tails = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5]])
        assert study.frontier_sets(models, medians, tails) == [("B", "A"), ("A",)]

    def test_membership_frequency_matches_bootstrap_frontier_probability(self):
        # Identical inputs, with ties on both axes, through both code paths.
        rng = np.random.default_rng(0)
        models = [f"m{i}" for i in range(8)]
        medians = rng.integers(0, 4, size=(40, 8)).astype(float)
        tails = rng.integers(0, 4, size=(40, 8)).astype(float)
        sets = study.frontier_sets(models, medians, tails)
        assert _frequencies(models, sets) == bootstrap.frontier_probability(models, medians, tails)

    def test_the_cross_benchmark_frontier_combines_replicate_r_of_every_cohort(self):
        models = ["A", "B", "C"]
        # Replicate 0. X: medians A > B > C, tails C > A > B; Y: medians B > C > A, tails
        # C > B > A. Mean median ranks A 2, B 1.5, C 2.5; mean tail ranks A 2.5, B 2.5, C 1:
        # B and C are undominated. Replicate 1: A is highest everywhere, so A alone.
        x = (
            np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
            np.array([[2.0, 1.0, 3.0], [3.0, 2.0, 1.0]]),
        )
        y = (
            np.array([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]]),
            np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        )
        sets = study.mean_rank_frontier_sets(models, {"X": x, "Y": y})
        assert [set(s) for s in sets] == [{"B", "C"}, {"A"}]

    def test_cross_benchmark_frequency_matches_bootstrap_mean_rank_frontier(self):
        rng = np.random.default_rng(1)
        models = [f"m{i}" for i in range(6)]
        cohorts = {
            c: tuple(rng.integers(0, 3, size=(30, 6)).astype(float) for _ in range(2))
            for c in ("X", "Y", "Z")
        }
        sets = study.mean_rank_frontier_sets(models, cohorts)
        expected = bootstrap.mean_rank_frontier_probability(models, cohorts)
        assert _frequencies(models, sets) == expected


class TestPairedMembership:
    def test_arms_are_compared_replicate_by_replicate_in_steps_of_one_replicate(self):
        models = ["A", "B", "C"]
        fixed = [("A",), ("A", "B"), ("A",), ("A",)]
        recomputed = [("B", "A"), ("A",), ("B", "A"), ("C",)]
        table = study.paired_membership(models, fixed, recomputed).set_index("model")
        assert table.loc["A", ["hits_fixed", "hits_recomputed"]].tolist() == [4, 3]
        assert table.loc["B", ["freq_fixed", "freq_recomputed", "freq_diff"]].tolist() == [
            0.25,
            0.5,
            0.25,
        ]
        # B: replicate 1 fixed only; replicates 0 and 2 recomputed only.
        assert table.loc["B", ["n_fixed_only", "n_recomputed_only"]].tolist() == [1, 2]
        assert table.loc["C", "freq_diff"] == 0.25
        assert (table["n_replicates"] == 4).all() and (table["resolution"] == 0.25).all()

    def test_only_differences_of_more_than_one_replicate_step_are_flagged(self):
        models = ["A", "B", "C", "D"]
        fixed = [("A",), ("A",), ("A",), ("A", "D")]
        recomputed = [("B",), ("B",), ("A",), ("A", "C")]
        table = study.paired_membership(models, fixed, recomputed)
        # A loses two replicates and B gains two: flagged. C gains one, D loses one: a single
        # step is the resolution, not a difference.
        assert study.beyond_one_step(table)["model"].tolist() == ["A", "B"]


# --------------------------------------------------------------------------- the whole study

CONTROL = "DINOv2-B"
RANKED = ("A", "B", "C")
#: Each encoder's margins sit in its own band, so A beats B beats C on both axes wherever
#: the fixed arm looks: its frontier is A alone in every replicate. Inside the band a row's
#: margin depends on its unit, so the fixed arm's medians do depend on which units it reads.
BANDS = {"A": 0.6, "B": 0.3, "C": 0.0, CONTROL: 0.9}
FRACTION = 25
#: The recomputed arm, hand-chosen, identical on every cohort: (median, LTM_10) per encoder.
#:   r0: B has the mildest tail, A the highest median -> frontier {B, A}; median order A B C.
#:   r1: A is highest on both -> frontier {A}; median order A C B.
RECOMPUTED = [
    {"A": (0.5, -0.2), "B": (0.4, -0.1), "C": (0.1, -0.3), CONTROL: (0.9, 0.9)},
    {"A": (0.5, -0.1), "B": (0.2, -0.2), "C": (0.3, -0.3), CONTROL: (0.9, 0.9)},
]
SHAPES = {
    "pathorob-camelyon": False,
    "pathorob-tcga-4x4": False,
    "pathorob-tolkach-esca": True,
}


def _margins(manifest: pd.DataFrame, model: str) -> np.ndarray:
    unit = manifest[["label", "medical_center", "group_id"]].astype(str).agg("|".join, axis=1)
    offsets = unit.map(lambda key: (sum(map(ord, key)) % 50) / 1000.0)
    tile = manifest["sample_id"].str.endswith("t1").astype(float) * 0.01
    return BANDS[model] + offsets.to_numpy() + tile.to_numpy()


def _scores_frame(values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    models = [*RANKED, CONTROL]
    return pd.DataFrame(
        {
            "model": models,
            "croma": [values[m][0] for m in models],
            "croma_f0": 0.0,
            "croma_ltm_alpha": [values[m][1] for m in models],
        }
    )


def _meta(benchmark: str, manifest: pd.DataFrame) -> dict:
    return {
        "benchmark": benchmark,
        "n_rows": int(len(manifest)),
        "n_units": int(len(manifest[list(subsampling.UNIT)].drop_duplicates())),
        "roster": [*RANKED, CONTROL],
    }


def _build_tree(tmp_path: Path) -> dict:
    """Production runs, manifests and a one-fraction, two-replicate subsampling study."""
    from benchmarks import get as benchmark_spec
    from croma.metrics.tail import compute_tail_metrics
    from croma.metrics.croma import CROMA_HEADLINE_M
    from paper_manifest import by_benchmark

    root = tmp_path / "repo"
    sub_root = root / "output" / "studies" / "subsampling-stability"
    design = subsampling.design(fractions=(FRACTION,), n_replicates=len(RECOMPUTED))
    subsampling.write_design(sub_root, design)
    drawn = {}
    for benchmark, shared in SHAPES.items():
        full, _ = subsampling.UNITS_PER_CELL[benchmark]
        manifest = _cohort_manifest(classes=2, centres=2, units_per_cell=full, shared_cases=shared)
        path = root / benchmark_spec(benchmark).manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.assign(image_path="x.png").to_csv(path, index=False)

        entry = by_benchmark(benchmark)
        per_sample, published = [], []
        for model in (*RANKED, CONTROL):
            margins = _margins(manifest, model)
            per_sample.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "occurrence_index": np.arange(len(manifest)),
                        "sample_index": np.arange(len(manifest)),
                        "subset": "dataset",
                        "sample_id": manifest["sample_id"],
                        "group_id": manifest["group_id"],
                        "label": manifest["label"],
                        "confounder": manifest["medical_center"],
                        f"croma_m{int(CROMA_HEADLINE_M)}": margins,
                    }
                )
            )
            published.append(
                {
                    "model": model,
                    "croma": float(np.median(margins)),
                    "croma_f0": 0.0,
                    "croma_ltm_alpha": compute_tail_metrics(margins, alpha=0.1).ltm_alpha,
                }
            )
        results = root / entry.metrics_rel
        results.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(per_sample).to_csv(root / entry.per_sample_rel, index=False)
        published = pd.DataFrame(published)
        published.to_csv(results, index=False)

        full_dir = subsampling.replicate_dir(sub_root, benchmark, subsampling.FULL, 0)
        full_scores = published.set_index("model")[["croma", "croma_ltm_alpha"]]
        subsampling.record_replicate(
            full_dir,
            _scores_frame({m: tuple(full_scores.loc[m]) for m in full_scores.index}),
            _meta(benchmark, manifest),
        )
        manifest[list(subsampling.UNIT)].drop_duplicates().to_csv(
            full_dir / subsampling.UNITS, index=False
        )
        (full_dir / subsampling.REPRODUCTION).write_text('{"identical": true}\n')
        for replicate, values in enumerate(RECOMPUTED):
            out = subsampling.replicate_dir(sub_root, benchmark, FRACTION, replicate)
            sample = subsampling.draw_subsample(manifest, benchmark, FRACTION, replicate)
            subsampling.record_replicate(out, _scores_frame(values), _meta(benchmark, sample))
            sample[list(subsampling.UNIT)].drop_duplicates().to_csv(
                out / subsampling.UNITS, index=False
            )
            drawn[(benchmark, replicate)] = sample
    return {"root": root, "sub_root": sub_root, "design": design, "drawn": drawn}


@pytest.fixture
def tree(tmp_path):
    return _build_tree(tmp_path)


def _read_manifest(tree, benchmark: str) -> pd.DataFrame:
    from benchmarks import get as benchmark_spec

    return pd.read_csv(tree["root"] / benchmark_spec(benchmark).manifest)


def _run(tree, out):
    return study.run(
        root=tree["root"],
        subsampling_root=tree["sub_root"],
        out_dir=out,
        expected_design=tree["design"],
    )


class TestStudy:
    def test_membership_is_written_per_cohort_fraction_and_ranked_encoder(self, tree, tmp_path):
        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        table = pd.read_csv(out / "membership.csv")
        assert set(table["cohort"]) == set(SHAPES)
        assert set(table["fraction"]) == {FRACTION}
        assert sorted(set(table["model"])) == list(RANKED)  # the control is not ranked
        for cohort in SHAPES:
            rows = table[table["cohort"] == cohort].set_index("model")
            assert rows.loc["A", ["freq_fixed", "freq_recomputed"]].tolist() == [1.0, 1.0]
            assert rows.loc["B", ["freq_fixed", "freq_recomputed", "freq_diff"]].tolist() == [
                0.0,
                0.5,
                0.5,
            ]
            assert rows.loc["C", "freq_recomputed"] == 0.0
            assert (rows["resolution"] == 0.5).all()
            # Each arm's spread across replicates: the recomputed medians are the literals.
            assert rows.loc["C", ["recomputed_median_min", "recomputed_median_max"]].tolist() == [
                0.1,
                0.3,
            ]
            assert rows.loc["B", "recomputed_ltm_mean"] == pytest.approx(-0.15)

    def test_the_fixed_arm_summarises_each_replicates_drawn_rows(self, tree, tmp_path):
        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        scores = pd.read_csv(out / "replicate_scores.csv", float_precision="round_trip")
        for (cohort, replicate), sample in tree["drawn"].items():
            for model in RANKED:
                row = scores[
                    (scores["cohort"] == cohort)
                    & (scores["replicate"] == replicate)
                    & (scores["model"] == model)
                ]
                assert len(row) == 1
                expected = np.median(_margins(sample, model))
                assert row["fixed_median"].item() == pytest.approx(expected, abs=1e-15)
                assert row["n_anchors"].item() == len(sample)
                assert row["recomputed_median"].item() == RECOMPUTED[replicate][model][0]

    def test_both_arms_reproduce_the_production_frontier_on_the_full_cohort(self, tree, tmp_path):
        import json

        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        record = json.loads((out / "reproduction.json").read_text())
        for cohort in SHAPES:
            entry = record["cohorts"][cohort]
            assert entry["production_frontier"] == ["A"]
            assert entry["fixed_frontier"] == entry["recomputed_frontier"] == ["A"]
            assert entry["fixed_max_abs_deviation"] == {"median": 0.0, "ltm": 0.0}
            assert entry["n_anchors"] == len(_read_manifest(tree, cohort))
        points = pd.read_csv(out / "full_cohort_scores.csv", float_precision="round_trip")
        assert len(points) == len(SHAPES) * len(RANKED)
        assert (points["fixed_median"] == points["production_median"]).all()
        assert (points["recomputed_ltm"] == points["production_ltm"]).all()

    def test_margins_that_do_not_reproduce_the_published_scores_are_refused(self, tree, tmp_path):
        from paper_manifest import by_benchmark

        path = tree["root"] / by_benchmark("pathorob-tcga-4x4").metrics_rel
        published = pd.read_csv(path)
        published.loc[published["model"] == "B", "croma_ltm_alpha"] += 1e-9
        published.to_csv(path, index=False)
        with pytest.raises(study.ReproductionError, match="tcga-4x4.*B"):
            _run(tree, tmp_path / "frontier-subsampling")

    def test_a_replicate_whose_anchors_differ_from_the_rows_it_scored_is_refused(
        self, tree, tmp_path
    ):
        import json

        where = subsampling.replicate_dir(tree["sub_root"], "pathorob-camelyon", FRACTION, 1)
        meta = json.loads((where / subsampling.REPLICATE_META).read_text())
        meta["n_rows"] += 1
        (where / subsampling.REPLICATE_META).write_text(json.dumps(meta))
        with pytest.raises(study.ReproductionError, match="camelyon.*25%.*replicate 1"):
            _run(tree, tmp_path / "frontier-subsampling")

    @pytest.mark.parametrize("dropped", [1, 2], ids=["a unit", "a whole cell"])
    def test_a_draw_that_breaks_the_stated_units_per_cell_is_refused(self, tree, tmp_path, dropped):
        import json

        where = subsampling.replicate_dir(tree["sub_root"], "pathorob-tolkach-esca", FRACTION, 0)
        units = pd.read_csv(where / subsampling.UNITS, dtype=str)
        # Tolkach keeps two units per cell at 25%: drop one of a cell's, or both.
        first_cell = (
            units[["label", "medical_center"]] == units.iloc[0][["label", "medical_center"]]
        ).all(axis=1)
        units = units.drop(units.index[first_cell][:dropped])
        units.to_csv(where / subsampling.UNITS, index=False)
        meta = json.loads((where / subsampling.REPLICATE_META).read_text())
        manifest = _read_manifest(tree, "pathorob-tolkach-esca").astype(str)
        meta["n_rows"] = int(len(study.retained_rows(manifest, units)))
        meta["n_units"] = int(len(units))
        (where / subsampling.REPLICATE_META).write_text(json.dumps(meta))
        with pytest.raises(study.ReproductionError, match="tolkach.*units per cell"):
            _run(tree, tmp_path / "frontier-subsampling")

    def test_the_cross_benchmark_frontier_pairs_replicate_r_of_every_cohort(self, tree, tmp_path):
        import json

        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        table = pd.read_csv(out / "cross_benchmark_membership.csv").set_index("model")
        assert table.index.tolist() == list(RANKED)
        assert (table["fraction"] == FRACTION).all()
        # Fixed: A first on everything, everywhere. Recomputed r0: every cohort ranks medians
        # A B C and tails B A C, so A and B tie on mean rank sum and are both undominated.
        assert table.loc["A", ["freq_fixed", "freq_recomputed"]].tolist() == [1.0, 1.0]
        assert table.loc["B", ["freq_fixed", "freq_recomputed"]].tolist() == [0.0, 0.5]
        full = json.loads((out / "reproduction.json").read_text())["cross_benchmark"]
        assert full == {
            "production_frontier": ["A"],
            "fixed_frontier": ["A"],
            "recomputed_frontier": ["A"],
        }

    def test_spearman_rho_compares_the_arms_median_rankings_per_replicate(self, tree, tmp_path):
        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        rho = pd.read_csv(out / "rank_agreement.csv")
        assert len(rho) == len(SHAPES) * len(RECOMPUTED)
        assert (rho["n_models"] == len(RANKED)).all()
        # Fixed: A > B > C. Recomputed r0 keeps that order; r1 swaps B and C, so
        # rho = 1 - 6 * (0 + 1 + 1) / (3 * (9 - 1)) = 0.5.
        by_replicate = rho.groupby("replicate")["spearman_rho"].unique()
        assert by_replicate.loc[0].tolist() == [1.0]
        assert by_replicate.loc[1].tolist() == [0.5]

    def test_every_replicate_frontier_set_is_written_for_both_arms(self, tree, tmp_path):
        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        sets = pd.read_csv(out / "frontier_sets.csv", keep_default_na=False)
        assert set(sets["scope"]) == {*SHAPES, "cross-benchmark"}
        assert len(sets) == (len(SHAPES) + 1) * len(RECOMPUTED) * 2
        row = sets.set_index(["scope", "fraction", "replicate", "arm"])
        assert row.loc[("pathorob-camelyon", FRACTION, 0, "recomputed"), "frontier"] == "B;A"
        assert row.loc[("pathorob-camelyon", FRACTION, 0, "fixed"), "frontier"] == "A"
        assert row.loc[("cross-benchmark", FRACTION, 1, "recomputed"), "n_frontier"] == 1

    def test_the_units_each_replicate_retained_are_written_per_cell(self, tree, tmp_path):
        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        cells = pd.read_csv(out / "units_per_cell.csv")
        for cohort in SHAPES:
            mine = cells[cells["cohort"] == cohort]
            assert len(mine) == 4 * len(RECOMPUTED)  # 2 classes x 2 centres per replicate
            stated = subsampling.units_per_cell(cohort, FRACTION)
            assert (mine["n_units"] == stated).all()
            assert (mine["n_anchors"] == 2 * stated).all()  # two tiles per unit

    def test_the_summary_labels_frequencies_with_their_resolution(self, tree, tmp_path):
        import json

        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        summary = json.loads((out / "summary.json").read_text())
        assert "membership frequency under subsampling" in summary["measure"]
        assert "not a bootstrap probability" in summary["measure"]
        assert summary["n_replicates"] == 2 and summary["resolution"] == 0.5
        assert summary["alpha"] == 0.1
        # B differs between the arms by exactly one replicate: one step is the resolution.
        assert summary["beyond_one_step"] == []
        agreement = summary["rank_agreement"]["pathorob-camelyon"][str(FRACTION)]
        assert agreement == {"median": 0.75, "min": 0.5, "max": 1.0}

    def test_provenance_records_the_sha256_of_every_input_and_output(self, tree, tmp_path):
        import hashlib
        import json

        from benchmarks import get as benchmark_spec
        from paper_manifest import by_benchmark

        out = tmp_path / "frontier-subsampling"
        _run(tree, out)
        provenance = json.loads((out / "provenance.json").read_text())
        root, sub_root = tree["root"], tree["sub_root"]
        expected = [sub_root / "design.json"]
        for cohort in SHAPES:
            entry = by_benchmark(cohort)
            expected += [root / entry.per_sample_rel, root / entry.metrics_rel]
            expected.append(root / benchmark_spec(cohort).manifest)
            full = subsampling.replicate_dir(sub_root, cohort, subsampling.FULL, 0)
            expected += [full / "scores.csv", full / "replicate.json", full / "reproduction.json"]
            for replicate in range(len(RECOMPUTED)):
                where = subsampling.replicate_dir(sub_root, cohort, FRACTION, replicate)
                expected += [where / "scores.csv", where / "replicate.json", where / "units.csv"]
        digest = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in expected
        }
        assert provenance["inputs"] == digest
        written = sorted(p.name for p in out.iterdir() if p.name != "provenance.json")
        assert sorted(provenance["outputs"]) == written
        for name, sha in provenance["outputs"].items():
            assert sha == hashlib.sha256((out / name).read_bytes()).hexdigest()
