"""The probe-record reductions' run and paper surfaces: the study run, its basis,
``tab:low-data``, the sensitivity and specificity rows of ``tab:nipd-auroc``, and macros.

The study (``scripts/studies/probe_reductions.py``) reads each cohort's CRoMa from the run
``paper_manifest`` resolves, so its end-to-end run is tested here, on a checkout-shaped
``tmp_path`` tree, rather than in ``test_probe_reductions_study.py``. The sdist leaves this
module out together with ``scripts/repro``. No ``paper/`` or ``output/`` tree is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (
    ROOT / "scripts" / "repro",
    ROOT / "scripts" / "bench",
    ROOT / "scripts" / "studies",
    ROOT / "scripts" / "studies" / "apd",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import probe_reductions as study  # noqa: E402

RANKED = ["E0", "E1", "E2", "E3"]
CONTROL = "DINOv2-B"
CROMA = {"E0": 0.1, "E1": 0.2, "E2": 0.3, "E3": 0.4, CONTROL: -0.2}
ITERATIONS = 2
#: Small resampling counts keep the run fast; the outputs must report whichever were used.
FAST = dict(n_boot=40, n_permutations=60)


def _camelyon_scores(index: int, n_splits: int):
    """Encoder ``index``'s (sensitivity, specificity) per (split, replicate), both arms.

    Every probe starts perfect. Sensitivity halves from split ``4 + index`` on and
    specificity from split ``5 + index`` on, so the least robust encoder (E0, lowest CRoMa)
    loses both first, and E3 keeps its specificity. Out of domain the same, except that E3's
    specificity is stuck at 0.5 -- no skill above chance at V = 0.
    """
    split = np.arange(n_splits)[:, None] * np.ones((1, ITERATIONS))
    sens = np.where(split < 4 + index, 1.0, 0.5)
    spec = np.where(split < 5 + index, 1.0, 0.5)
    ood_spec = np.full_like(spec, 0.5) if index == 3 else spec
    return {"in_domain": (sens, spec), "out_of_domain": (sens, ood_spec)}


def _margins(sens: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Margins over two tumour then two normal rows that score ``sens`` and ``spec``."""
    margins = np.empty((*sens.shape, 4, 1))
    margins[..., 0, 0] = np.where(sens == 1.0, 1.0, -1.0)
    margins[..., 1, 0] = 1.0
    margins[..., 2, 0] = np.where(spec == 1.0, -1.0, 1.0)
    margins[..., 3, 0] = -1.0
    return margins


def _write_record(path: Path, scores: dict, *, tiles: int, v: list[float], shift: float = 0.0):
    """A camelyon record of eight in-domain rows (even rows tumour) and four OOD rows."""
    # Each replicate holds out two tumour rows, then two normal rows: rows 6 and 7 never.
    held_out = np.array([[0, 2, 1, 3], [2, 4, 3, 5]])
    id_label = np.array(["tumor" if r % 2 == 0 else "normal" for r in range(8)])
    arrays = {
        "meta": json.dumps(
            {
                "biological_classes": ["normal", "tumor"],
                "cramers_v": list(v),
                "n_splits": len(v),
                "tiles_per_slide": tiles,
                "iterations": ITERATIONS,
            }
        ),
        "held_out": held_out,
        "classes": np.array([0, 1]),
        "id_sample_id": np.array([f"s{r}" for r in range(8)]),
        "id_group_id": np.array([f"g{r // 2}" for r in range(8)]),
        "id_label": id_label,
        "id_confounder": np.array(["RUMC"] * 4 + ["UMCU"] * 4),
        "ood_label": np.array(["tumor", "tumor", "normal", "normal"]),
    }
    for name, (sens, spec) in scores.items():
        arrays[f"margins_{name}"] = _margins(sens, spec)
        arrays[f"accuracies_{name}"] = (sens + spec) / 2 - shift
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    """A checkout-shaped tree for Camelyon at 300 and 3 tiles per slide.

    The full-tile APD cell holds the full-tile record's accuracies. The 3-tile records'
    balanced accuracies sit 0.1 lower at every probe.
    """
    from loaders import training_correlations
    from paper_manifest import by_benchmark

    root = tmp_path_factory.mktemp("checkout")
    v = training_correlations("camelyon")
    run = by_benchmark("pathorob-camelyon")
    models = [*RANKED, CONTROL]
    for i, model in enumerate(models):
        scores = _camelyon_scores(min(i, 3), len(v))
        for tiles, shift in ((300, 0.0), (3, 0.1)):
            path = root / study.RECORDS_REL / "camelyon" / f"tiles-{tiles}" / f"{model}.npz"
            _write_record(path, scores, tiles=tiles, v=v, shift=shift)
        cell = root / study.APD_REL / "camelyon" / f"{model}.json"
        cell.parent.mkdir(parents=True, exist_ok=True)
        cell.write_text(
            json.dumps(
                {
                    f"{arm}_test_accuracies": ((s[0] + s[1]) / 2).tolist()
                    for arm, s in zip(("id", "ood"), scores.values())
                }
            )
        )
    metrics = root / run.metrics_rel
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"model": models, "croma": [CROMA[m] for m in models], "ri": 0.5, "mari": 0.5, "k": 11}
    ).to_csv(metrics, index=False)
    replicates = root / run.studies_rel / "bootstrap_replicates.npz"
    replicates.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    medians = np.array([CROMA[m] for m in models]) + rng.normal(0, 0.01, (FAST["n_boot"], 5))
    np.savez(replicates, models=np.array(models), median=medians)
    study.run(root, runs={"camelyon": (300, 3)}, **FAST)
    return root


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / study.STUDY_REL / name, keep_default_na=False, na_values=[""])


class TestRun:
    def test_coverage_counts_the_rows_never_held_out(self, tree):
        coverage = _read(tree, "coverage.csv").set_index("cohort")

        assert coverage.loc["camelyon", "rows"] == 8
        assert coverage.loc["camelyon", "never_held_out"] == 2
        assert coverage.loc["camelyon", "share_never_held_out"] == pytest.approx(0.25)
        counts = _read(tree, "held_out_counts.csv")
        assert counts["times_held_out"].tolist() == [1, 1, 2, 2, 1, 1, 0, 0]
        assert counts["sample_id"].tolist() == [f"s{r}" for r in range(8)]

    def test_every_probe_is_scored_for_sensitivity_and_specificity(self, tree):
        probes = _read(tree, "sens_spec_probes.csv")

        # Five encoders x two regimes x eight splits x two replicates.
        assert len(probes) == 5 * 2 * 8 * 2
        e0 = probes[(probes["model"] == "E0") & (probes["regime"] == "id")]
        assert e0.groupby("split")["sensitivity"].mean().tolist() == [1.0] * 4 + [0.5] * 4

    def test_a_specificity_without_skill_at_balance_has_no_nipd(self, tree):
        sens_spec = _read(tree, "sens_spec.csv").set_index(["model", "regime"])

        assert np.isnan(sens_spec.loc[("E3", "ood"), "nipd_spec"])
        assert sens_spec.loc[("E3", "ood"), "spec_v0"] == pytest.approx(0.5)
        # Its sensitivity, and its in-domain specificity, are defined.
        assert np.isfinite(sens_spec.loc[("E3", "ood"), "nipd_sens"])
        assert sens_spec.loc[("E3", "id"), "nipd_spec"] == pytest.approx(0.0)

    def test_sensitivity_and_specificity_rhos_form_the_tile_cohorts_family(self, tree):
        corr = _read(tree, "correlations.csv")
        rows = corr[corr["analysis"] == "sens_spec"].set_index(["regime", "target"])

        # The less robust the encoder, the earlier it loses sensitivity and specificity.
        assert rows.loc[("id", "nipd_sens"), "rho"] == pytest.approx(1.0)
        assert rows.loc[("id", "nipd_spec"), "rho"] == pytest.approx(1.0)
        assert rows.loc[("id", "nipd_sens"), "n"] == 4
        # E3 drops out of the out-of-domain specificity correlation, and is named.
        assert rows.loc[("ood", "nipd_spec"), "n"] == 3
        assert rows.loc[("ood", "nipd_spec"), "undefined"] == "E3"
        assert rows["in_family"].all() and rows["q_value"].notna().all()
        assert set(rows["n_boot"]) == {FAST["n_boot"]}

    def test_the_low_data_summary_reads_the_full_level_from_the_apd_cell(self, tree):
        low = _read(tree, "low_data.csv").set_index(["tiles", "model"])

        assert sorted(set(low.index.get_level_values("tiles"))) == [3, 300]
        assert len(low) == 2 * 5
        assert low.loc[(300, "E0"), "bacc_v0_id"] == pytest.approx(1.0)
        assert low.loc[(3, "E0"), "bacc_v0_id"] == pytest.approx(0.9)
        assert bool(low.loc[(300, "E0"), "full_tiles"]) and not bool(
            low.loc[(3, "E0"), "full_tiles"]
        )
        assert bool(low.loc[(3, CONTROL), "control"])

    def test_low_data_rhos_carry_both_intervals_and_one_family(self, tree):
        corr = _read(tree, "correlations.csv")
        rows = corr[corr["analysis"] == "low_data"]

        assert sorted(zip(rows["tiles"], rows["regime"])) == [
            (3, "id"),
            (3, "ood"),
            (300, "id"),
            (300, "ood"),
        ]
        assert (rows["n"] == 4).all()
        assert (
            rows[["ci_lo", "ci_hi", "two_level_lo", "two_level_hi", "q_value"]].notna().all().all()
        )
        assert rows["in_family"].all()

    def test_the_study_records_the_sha256_of_what_it_read_and_wrote(self, tree):
        info = json.loads((tree / study.STUDY_REL / "study.json").read_text())

        assert set(info["outputs"]) == set(study.OUTPUTS)
        assert "output/studies/probe-records/camelyon/tiles-3/E0.npz" in info["inputs"]
        assert "output/studies/apd/camelyon/E0.json" in info["inputs"]
        assert info["low_data_cohorts"] == ["camelyon"]

    def test_a_missing_record_fails_the_run(self, tree, tmp_path):
        with pytest.raises(FileNotFoundError, match="tiles-30"):
            study.run(tree, tmp_path, runs={"camelyon": (300, 30)}, **FAST)


def _copy_study(tree: Path, root: Path) -> Path:
    """``tree``'s study outputs and the inputs they fingerprint, copied under ``root``."""
    import shutil

    info = json.loads((tree / study.STUDY_REL / "study.json").read_text())
    for relative in [*info["inputs"], *[f"{study.STUDY_REL}/{n}" for n in info["outputs"]]]:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tree / relative, root / relative)
    shutil.copyfile(tree / study.STUDY_REL / "study.json", root / study.STUDY_REL / "study.json")
    return root


class TestBasis:
    def test_a_study_never_run_fails_loudly(self, tmp_path):
        import _probe_reductions as basis

        with pytest.raises(FileNotFoundError, match="probe_reductions.py"):
            basis.load(tmp_path)

    def test_a_changed_output_is_refused(self, tree, tmp_path):
        import _probe_reductions as basis
        from _paper_tables import CaptionClaimError

        root = _copy_study(tree, tmp_path)
        low_data = root / study.STUDY_REL / "low_data.csv"
        low_data.write_text(low_data.read_text().replace("E0", "E9"))

        with pytest.raises(CaptionClaimError, match="low_data.csv"):
            basis.load(root)

    def test_a_changed_input_is_refused(self, tree, tmp_path):
        import _probe_reductions as basis
        from _paper_tables import CaptionClaimError

        root = _copy_study(tree, tmp_path)
        cell = root / study.APD_REL / "camelyon" / "E1.json"
        cell.write_text(cell.read_text() + " ")

        with pytest.raises(CaptionClaimError, match="E1.json"):
            basis.load(root)

    def test_a_current_study_loads_its_tables(self, tree):
        import _probe_reductions as basis

        loaded = basis.load(tree)

        assert loaded.low_data_cohorts == ["camelyon"]
        assert loaded.tile_levels("camelyon") == [300, 3]
        assert len(loaded.sens_spec_correlations) == 4
        assert loaded.n_ranked == 4


def _planted(reductions, *, low_data=None, correlations=None):
    """``reductions`` with chosen low-data and correlation cells replaced by known values.

    ``low_data`` maps (tiles, model) to columns; ``correlations`` maps (analysis, cohort,
    tiles, regime, target) to columns.
    """
    import _probe_reductions as basis

    low, corr = reductions.low_data.copy(), reductions.correlations.copy()
    for (tiles, model), values in (low_data or {}).items():
        mask = (low["tiles"] == tiles) & (low["model"] == model)
        assert mask.sum() == 1
        for column, value in values.items():
            low.loc[mask, column] = value
    for key, values in (correlations or {}).items():
        mask = np.logical_and.reduce(
            [corr[c] == v for c, v in zip(("analysis", "cohort", "tiles", "regime", "target"), key)]
        )
        assert mask.sum() == 1
        for column, value in values.items():
            corr.loc[mask, column] = value
    return basis.ProbeReductions(
        info=reductions.info,
        coverage=reductions.coverage,
        sens_spec=reductions.sens_spec,
        low_data=low,
        correlations=corr,
    )


def _line(tex: str, lead: str) -> list[str]:
    line = next(line for line in tex.splitlines() if line.startswith(lead))
    return line.removesuffix(r" \\").split(" & ")


class TestLowDataTable:
    @pytest.fixture
    def reductions(self, tree):
        import _probe_reductions as basis

        return basis.load(tree)

    def test_one_block_per_cohort_and_one_row_per_tile_level(self, reductions):
        import generate_low_data_table as table

        tex = table.build(reductions)

        assert tex.startswith("% AUTO-GENERATED")
        assert r"\label{tab:low-data}" in tex
        assert r"\multicolumn{5}{l}{\emph{Camelyon}} \\" in tex
        assert len(_line(tex, "300 (full)")) == 5
        assert len(_line(tex, "3 &")) == 5

    def test_a_row_gives_the_balanced_accuracy_median_and_range_and_both_rhos(self, reductions):
        import generate_low_data_table as table

        planted = _planted(
            reductions,
            low_data={
                (3, "E0"): dict(bacc_v0_id=0.80),
                (3, "E1"): dict(bacc_v0_id=0.84),
                (3, "E2"): dict(bacc_v0_id=0.90),
                (3, "E3"): dict(bacc_v0_id=0.99),
                # The control is a floor, not a competitor: it never enters the range.
                (3, CONTROL): dict(bacc_v0_id=0.10),
            },
            correlations={
                ("low_data", "camelyon", 3, "id", "nipd_id"): dict(
                    rho=0.920684,
                    ci_lo=0.806669,
                    ci_hi=0.963487,
                    two_level_lo=0.768968,
                    two_level_hi=0.944361,
                ),
            },
        )

        encoder = _line(table.build(planted, design="encoder"), "3 &")
        two_level = _line(table.build(planted, design="two-level"), "3 &")

        assert encoder[1] == r"$0.87$\,{\scriptsize$[0.80, 0.99]$}"
        assert encoder[3] == r"$0.92$\,{\scriptsize$[0.81, 0.96]$}"
        assert two_level[3] == r"$0.92$\,{\scriptsize$[0.77, 0.94]$}"

    def test_the_caption_quotes_the_panel_and_the_interval_design(self, reductions):
        import generate_low_data_table as table

        encoder = table.build(reductions, design="encoder").split(r"\caption{")[1]
        two_level = table.build(reductions, design="two-level").split(r"\caption{")[1]

        assert "$4$ ranked encoders" in encoder
        assert f"${FAST['n_boot']}$" in encoder
        assert "encoder-bootstrap" in encoder and "probe runs" not in encoder
        assert "probe runs" in two_level
        # Two tile levels x two regimes on the one cohort: the q values' family. Four
        # encoders have 24 orderings, fewer than the permutations asked for, so p is exact.
        assert "Benjamini--Hochberg" in encoder and "$4$ associations" in encoder
        assert "Exact" in encoder and "all $24$ orderings" in encoder

    def test_a_monte_carlo_p_is_captioned_with_its_permutations(self, reductions):
        import _probe_reductions as basis
        import generate_low_data_table as table

        corr = reductions.correlations.copy()
        low = corr["analysis"] == "low_data"
        corr.loc[low, "permutation"] = "monte-carlo"
        corr.loc[low, "n_permutations"] = 10000
        drawn = basis.ProbeReductions(
            info=reductions.info,
            coverage=reductions.coverage,
            sens_spec=reductions.sens_spec,
            low_data=reductions.low_data,
            correlations=corr,
        )

        caption = table.build(drawn).split(r"\caption{")[1]

        assert "$10{,}000$ permutations" in caption and "orderings" not in caption

    def test_an_unknown_interval_design_is_refused(self, reductions):
        import generate_low_data_table as table

        with pytest.raises(ValueError, match="design"):
            table.build(reductions, design="slides")

    def test_the_table_rebuilds_byte_for_byte(self, reductions):
        import generate_low_data_table as table

        assert table.build(reductions) == table.build(reductions)


class TestMacros:
    def test_the_in_domain_rho_at_each_tile_level_per_cohort(self, tree):
        import _probe_reductions as basis
        import generate_low_data_table as table

        planted = _planted(
            basis.load(tree),
            correlations={
                ("low_data", "camelyon", 300, "id", "nipd_id"): dict(
                    rho=0.945983,
                    ci_lo=0.834398,
                    ci_hi=0.977351,
                    two_level_lo=0.820214,
                    two_level_hi=0.965878,
                    q_value=0.000129,
                ),
                ("low_data", "camelyon", 3, "id", "nipd_id"): dict(rho=0.675214, q_value=0.0234),
            },
        )

        encoder = table.macro_lines(planted, design="encoder")
        two_level = table.macro_lines(planted, design="two-level")

        assert r"\newcommand{\LowDataNModels}{4}" in encoder
        assert r"\newcommand{\LowDataIdCromaCamelyonTilesThreeHundred}{0.95}" in encoder
        assert r"\newcommand{\LowDataIdCromaCamelyonTilesThreeHundredCi}{$[0.83, 0.98]$}" in encoder
        assert (
            r"\newcommand{\LowDataIdCromaCamelyonTilesThreeHundredCi}{$[0.82, 0.97]$}" in two_level
        )
        assert r"\newcommand{\LowDataIdCromaCamelyonTilesThreeHundredQ}{$q{<}0.001$}" in encoder
        assert r"\newcommand{\LowDataIdCromaCamelyonTilesThree}{0.68}" in encoder
        assert r"\newcommand{\LowDataIdCromaCamelyonTilesThreeQ}{$q{=}0.023$}" in encoder
        # Only the in-domain rho: one value, interval and q per cohort x tile level.
        assert sum(r"\LowDataIdCroma" in line for line in encoder) == 2 * 3
        assert not [line for line in encoder if "Ood" in line]

    def test_macros_read_the_verified_study(self, tree, tmp_path):
        import generate_low_data_table as table

        assert len(table.macros(tree)) == 1 + 2 * 3
        with pytest.raises(FileNotFoundError, match="probe_reductions.py"):
            table.macros(tmp_path)

    def test_the_values_file_fails_without_the_study(self, tmp_path, monkeypatch):
        # Never warn-and-skip: a missing macro would compile to an undefined control
        # sequence at best and silently drop a sentence's number at worst.
        from types import SimpleNamespace

        import _exposure
        import _grade_matched
        import _mari_vs_ri
        import _nipd_auroc
        import _overlap
        import _panel_macros
        import _uncertainty
        import generate_paper_values as gpv
        import generate_sample_shortcut_floats
        import generate_supp_rank_table

        # The blocks before this one read other studies and live runs, some relative to the
        # repo rather than ``root``, and several raise when those are absent. Stub them so
        # the test sees only the missing probe reductions, on a clean clone too.
        monkeypatch.setattr(_mari_vs_ri, "load", lambda root: None)
        monkeypatch.setattr(_mari_vs_ri, "macros", lambda study: [])
        monkeypatch.setattr(gpv, "_load_rank_frontier_macros", lambda: [])
        monkeypatch.setattr(_panel_macros, "panel_macros", lambda root: [])
        monkeypatch.setattr(generate_supp_rank_table, "macros", lambda root: [])
        monkeypatch.setattr(gpv, "_load_finetune_pair_macros", lambda root: [])
        monkeypatch.setattr(generate_sample_shortcut_floats, "macros", lambda root: [])
        monkeypatch.setattr(_exposure, "build", lambda root: SimpleNamespace(macros=[]))
        monkeypatch.setattr(_overlap, "PER_SAMPLE", tmp_path / "absent.csv")
        monkeypatch.setattr(_uncertainty, "load", lambda root: None)
        monkeypatch.setattr(gpv, "_uncertainty_block", lambda study: [])
        monkeypatch.setattr(gpv, "_apd_block", lambda root: [])
        monkeypatch.setattr(_grade_matched, "load", lambda root: None)
        monkeypatch.setattr(gpv, "_grade_matched_macros", lambda study: [])
        monkeypatch.setattr(gpv, "_silhouette_block", lambda root, apd_path: [])
        monkeypatch.setattr(_nipd_auroc, "load", lambda root: None)
        monkeypatch.setattr(gpv, "_nipd_auroc_macros", lambda study: [])

        with pytest.raises(FileNotFoundError, match="probe_reductions.py"):
            gpv.build([], tmp_path, "auto")


def test_the_paper_build_never_skips_the_table():
    # build_paper skips a study generator whose required input is absent and reports the
    # build as OK; with no required input the generator always runs, so a missing or stale
    # study fails the build through _probe_reductions.load.
    import build_paper

    entries = [s for s in build_paper.STUDIES if s[0] == "generate_low_data_table.py"]

    assert len(entries) == 1
    assert entries[0][1] is None
    assert "probe_reductions.py" in entries[0][2]


def test_the_full_reproduction_reduces_the_records_after_writing_them():
    import reproduce_faithful

    labels = [label for label, _, _ in reproduce_faithful.STUDIES]
    scripts = {label: script for label, script, _ in reproduce_faithful.STUDIES}

    assert labels.index("bootstrap-uncertainty") < labels.index("probe-reductions")
    assert labels.index("probe-records") < labels.index("probe-reductions")
    assert scripts["probe-reductions"] == "scripts/studies/probe_reductions.py"
    assert "probe-reductions" not in reproduce_faithful.JOBS_STUDIES
