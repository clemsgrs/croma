"""The grade-matched SO-pool study's paper surfaces: its basis, ``tab:grade-matched``, macros.

Everything renders from a synthetic study directory under ``tmp_path``; no ``paper/`` or
``output/`` tree is needed. The study itself is tested in ``test_grade_matched_study.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "repro", ROOT / "scripts" / "bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _grade_matched as basis  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

ROSTER = ["MOOZY", "PRISM", "PRISM2", "Prov-GigaPath", "TITAN"]

#: Pooled (``grade == "all"``) summary rows, one per encoder.
_POOLED = {
    "MOOZY": dict(
        full_median=0.0429,
        full_f0=0.46,
        random_median=-0.0201,
        random_median_lo=-0.0412,
        random_median_hi=0.0013,
        random_f0=0.52,
        random_f0_lo=0.49,
        random_f0_hi=0.556,
        matched_median=0.0221,
        matched_f0=0.474,
        delta_grade=0.0422,
        p_one_sided=0.0,
        paired_shift_median=0.031,
        n_random_nonpositive=260,
        n_sign_change=52,
        sign_change_share=0.2,
    ),
    "PRISM": dict(
        full_median=0.3374,
        full_f0=0.2,
        random_median=0.2344,
        random_median_lo=0.2201,
        random_median_hi=0.2489,
        random_f0=0.25,
        random_f0_lo=0.23,
        random_f0_hi=0.27,
        matched_median=0.2591,
        matched_f0=0.22,
        delta_grade=0.0247,
        p_one_sided=0.005,
        paired_shift_median=0.02,
        n_random_nonpositive=120,
        n_sign_change=18,
        sign_change_share=0.15,
    ),
    "PRISM2": dict(
        full_median=0.4075,
        full_f0=0.15,
        random_median=0.3021,
        random_median_lo=0.29,
        random_median_hi=0.315,
        random_f0=0.19,
        random_f0_lo=0.17,
        random_f0_hi=0.21,
        matched_median=0.3300,
        matched_f0=0.17,
        delta_grade=0.0279,
        p_one_sided=0.0,
        paired_shift_median=0.025,
        n_random_nonpositive=95,
        n_sign_change=19,
        sign_change_share=0.2,
    ),
    "Prov-GigaPath": dict(
        full_median=-0.4241,
        full_f0=0.99,
        random_median=-0.5040,
        random_median_lo=-0.5131,
        random_median_hi=-0.4952,
        random_f0=0.996,
        random_f0_lo=0.992,
        random_f0_hi=1.0,
        matched_median=-0.5093,
        matched_f0=0.998,
        delta_grade=-0.0053,
        p_one_sided=0.665,
        paired_shift_median=-0.004,
        n_random_nonpositive=498,
        n_sign_change=0,
        sign_change_share=0.0,
    ),
    "TITAN": dict(
        full_median=-0.1584,
        full_f0=0.8,
        random_median=-0.2071,
        random_median_lo=-0.2188,
        random_median_hi=-0.1957,
        random_f0=0.85,
        random_f0_lo=0.83,
        random_f0_hi=0.87,
        matched_median=-0.2142,
        matched_f0=0.86,
        delta_grade=-0.0071,
        p_one_sided=0.9,
        paired_shift_median=-0.006,
        n_random_nonpositive=425,
        n_sign_change=4,
        sign_change_share=4 / 425,
    ),
}


def _write_study(
    root: Path,
    *,
    roster=ROSTER,
    per_grade: int = 50,
    run_rel: str | None = None,
) -> Path:
    study = root / basis.STUDY_REL
    study.mkdir(parents=True)
    # The run's per-sample file, whose fingerprint the study records.
    per_sample = root / by_benchmark("panda").per_sample_rel
    per_sample.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": list(roster), "croma_m5": 0.1}).to_csv(per_sample, index=False)
    rows = []
    for model in roster:
        rows.append({"model": model, "grade": "all", "n_anchors": 500, **_POOLED[model]})
        for grade in range(1, 6):
            rows.append({"model": model, "grade": str(grade), "n_anchors": 100, **_POOLED[model]})
    pd.DataFrame(rows).to_csv(study / "summary.csv", index=False)
    meta = {
        "benchmark": "panda",
        "run_rel": run_rel or by_benchmark("panda").run_rel,
        "per_sample_sha256": hashlib.sha256(per_sample.read_bytes()).hexdigest(),
        "roster": list(roster),
        "headline_m": 5,
        "seed": 0,
        "n_replicates": 200,
        "pool_size": 50,
        "cancer_slides_per_grade": {
            "karolinska": {str(g): per_grade for g in range(1, 6)},
            "radboud": {str(g): 50 for g in range(1, 6)},
        },
        "n_cancer_anchors": 500,
        "full_pool_reproduces_run": True,
    }
    (study / "meta.json").write_text(json.dumps(meta))
    return study


class TestBasis:
    def test_a_study_that_has_not_been_run_fails_loudly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="grade_matched_pools.py"):
            basis.load(tmp_path)

    def test_loads_one_pooled_row_per_encoder_under_its_published_name(self, tmp_path):
        _write_study(tmp_path)

        study = basis.load(tmp_path)

        assert len(study.pooled) == 5
        assert study.pooled.loc["PRISM2", "matched_median"] == pytest.approx(0.33)
        assert set(study.per_grade["grade"]) == {"1", "2", "3", "4", "5"}
        assert study.meta["n_replicates"] == 200

    def test_a_study_of_another_run_is_stale(self, tmp_path):
        # The paper's PCaBiop numbers come from the run paper_manifest names; a study that
        # read a different run would put two runs' numbers side by side.
        _write_study(tmp_path, run_rel="output/metrics/median-k/panda")

        with pytest.raises(CaptionClaimError, match="median-k"):
            basis.load(tmp_path)

    def test_a_rerun_of_the_benchmark_in_place_leaves_the_study_stale(self, tmp_path):
        # Re-running panda in place keeps its run_rel, so the path cannot tell; the study
        # records the fingerprint of the per-sample file its full-pool check read.
        _write_study(tmp_path)
        per_sample = tmp_path / by_benchmark("panda").per_sample_rel
        rerun = pd.concat(
            [pd.read_csv(per_sample), pd.DataFrame({"model": ["CHIEF"], "croma_m5": [0.2]})]
        )
        rerun.to_csv(per_sample, index=False)

        with pytest.raises(CaptionClaimError, match="per_sample_metrics.csv"):
            basis.load(tmp_path)

    def test_a_study_whose_run_lost_its_per_sample_file_is_stale(self, tmp_path):
        _write_study(tmp_path)
        (tmp_path / by_benchmark("panda").per_sample_rel).unlink()

        with pytest.raises(CaptionClaimError, match="per_sample_metrics.csv"):
            basis.load(tmp_path)

    def test_the_caption_balance_claim_is_checked_not_assumed(self, tmp_path):
        _write_study(tmp_path, per_grade=49)

        with pytest.raises(CaptionClaimError, match="50"):
            basis.load(tmp_path)

    def test_the_rendered_roster_is_the_runs(self, tmp_path):
        study_dir = _write_study(tmp_path)
        meta = json.loads((study_dir / "meta.json").read_text())
        meta["roster"] = ROSTER[:4]
        (study_dir / "meta.json").write_text(json.dumps(meta))

        with pytest.raises(CaptionClaimError, match="TITAN"):
            basis.load(tmp_path)


class TestTable:
    def _tex(self, tmp_path) -> str:
        import generate_grade_matched_table as table

        _write_study(tmp_path)
        return table.build(basis.load(tmp_path))

    def test_one_row_per_encoder_ordered_by_the_full_pool_median(self, tmp_path):
        tex = self._tex(tmp_path)

        names = ["PRISM2", "PRISM", "MOOZY", "TITAN", "Prov-GigaPath"]
        positions = [tex.index(f"\n{name} &") for name in names]
        assert positions == sorted(positions)
        assert r"\label{tab:grade-matched}" in tex

    def test_a_row_carries_full_random_range_and_matched_then_the_grade_effect(self, tmp_path):
        tex = self._tex(tmp_path)

        row = next(line for line in tex.splitlines() if line.startswith("Prov-GigaPath &"))
        cells = [c.strip() for c in row.removesuffix(r"\\").split("&")]
        assert cells[1:4] == [
            "$-0.424$",
            r"$-0.504$\,{\scriptsize$[-0.513, -0.495]$}",
            "$-0.509$",
        ]
        assert cells[4:7] == [
            "$0.990$",
            r"$0.996$\,{\scriptsize$[0.992, 1.000]$}",
            "$0.998$",
        ]
        # p is a share of 200 replicates, so its resolution is 0.005: three decimals.
        assert cells[7:] == ["$-0.005$", "$0.665$", r"0\%"]

    def test_a_p_of_zero_is_reported_as_below_one_replicate(self, tmp_path):
        tex = self._tex(tmp_path)

        row = next(line for line in tex.splitlines() if line.startswith("MOOZY &"))
        cells = [c.strip() for c in row.removesuffix(r"\\").split("&")]
        # 0 of 200 replicates reached the matched median: p < 1/200, not p = 0.
        assert cells[8] == "$<0.005$"
        assert cells[9] == r"20\%"

    def test_the_caption_states_the_design(self, tmp_path):
        tex = self._tex(tmp_path)
        caption = tex[tex.index(r"\caption{") :]

        assert "$50$ cancer slides per ISUP grade per centre" in caption
        assert "Only the SO side is matched" in caption
        assert "$200$ random replicates" in caption
        assert "seed $0$" in caption


class TestMacros:
    def test_grade_effect_and_pool_medians_per_encoder(self, tmp_path):
        import generate_paper_values as gpv

        _write_study(tmp_path)
        lines = gpv._grade_matched_macros(basis.load(tmp_path))

        assert r"\newcommand{\GradeMatchedPrismTwoDelta}{$0.028$}" in lines
        assert r"\newcommand{\GradeMatchedProvGigaPathMatchedMedian}{$-0.509$}" in lines
        assert r"\newcommand{\GradeMatchedProvGigaPathRandomMedian}{$-0.504$}" in lines
        assert r"\newcommand{\GradeMatchedMoozyDelta}{$0.042$}" in lines
        assert sum(r"\GradeMatchedTitan" in line for line in lines) == 3

    def test_grade_counts_per_provider(self, tmp_path):
        import generate_paper_values as gpv

        _write_study(tmp_path)
        lines = gpv._grade_matched_macros(basis.load(tmp_path))

        assert r"\newcommand{\GradeMatchedKarolinskaGradeOneSlides}{50}" in lines
        assert r"\newcommand{\GradeMatchedRadboudGradeFiveSlides}{50}" in lines
        assert sum("Slides}" in line for line in lines) == 10

    def test_the_values_file_fails_without_the_study(self, tmp_path, monkeypatch):
        import _overlap
        import generate_paper_values as gpv

        # Two blocks before this one read live runs relative to the repo, not ``root``;
        # stub them so the test sees only the missing study, on a clean clone too.
        monkeypatch.setattr(gpv, "_load_rank_frontier_change_macros", lambda: [])
        monkeypatch.setattr(_overlap, "PER_SAMPLE", tmp_path / "absent.csv")

        with pytest.raises(FileNotFoundError, match="grade_matched_pools.py"):
            gpv.build([], tmp_path, "auto")


class TestReproduction:
    """``reproduce_faithful.py`` must recompute ``panda`` in the study's numerical setting.

    The study asserts its full pool matches the run's ``croma_m5`` exactly, so a ``panda``
    recomputed under another BLAS kernel or thread count aborts the one-command run.
    """

    @staticmethod
    def _env_for(benchmark: str, monkeypatch) -> dict[str, str] | None:
        import reproduce_faithful as rf

        seen: list[dict[str, str] | None] = []
        monkeypatch.setattr(rf, "_replay_for", lambda entry, *, dry_run: [])
        monkeypatch.setattr(rf, "_run", lambda cmd, *, label, dry_run, env=None: seen.append(env))
        rf.run_benchmark(by_benchmark(benchmark), recompute=True, dry_run=True)
        (env,) = seen
        return env

    def test_the_panda_run_gets_the_grade_matched_study_env(self, monkeypatch):
        import reproduce_faithful as rf

        env = self._env_for("panda", monkeypatch)

        assert env is not None
        assert env == rf.STUDY_ENV["grade-matched"]

    @pytest.mark.parametrize("benchmark", ["panda-isup", "pathorob-camelyon", "prostate"])
    def test_other_runs_keep_the_caller_env(self, benchmark, monkeypatch):
        assert self._env_for(benchmark, monkeypatch) is None

    def test_the_dry_run_prints_the_panda_env(self, monkeypatch, capsys):
        import reproduce_faithful as rf

        monkeypatch.setattr(rf, "_replay_for", lambda entry, *, dry_run: [])
        rf.run_benchmark(by_benchmark("panda"), recompute=True, dry_run=True)

        out = capsys.readouterr().out
        assert out.startswith("[Panda] OPENBLAS_CORETYPE=SKYLAKEX OPENBLAS_NUM_THREADS=4 ")
