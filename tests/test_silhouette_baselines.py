"""The batch-mixing baselines study: batch-ASW and iLISI against CRoMa and in-domain nIPD.

``scripts/studies/silhouette_baselines.py`` scores every encoder with scib-metrics'
``silhouette_batch`` and ``ilisi_knn`` and relates the scores to median CRoMa and to nIPD
across the ranked panel. Two seams are tested here:

* the per-encoder scoring, against the reference package (skipped where scib-metrics is
  not installed, which is the default: it is a ``repro`` extra only);
* the across-encoder inference, which needs no reference package and always runs. Its
  expected values come from scipy's Spearman on directly drawn encoder resamples, not from
  the study's own arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import silhouette_baselines as study  # noqa: E402
from croma.metrics.bootstrap import benjamini_hochberg, encoder_bootstrap_draws  # noqa: E402

CONTROL = study.CONTROL_MODEL
COHORTS = ("camelyon", "tcga_4x4", "tolkach")
N_RANKED = 12


def _per_encoder(seed: int = 0) -> pd.DataFrame:
    """Three cohorts of twelve ranked encoders plus the control, loosely co-varying."""
    rng = np.random.default_rng(seed)
    frames = []
    for dataset in COHORTS:
        latent = rng.normal(size=N_RANKED)
        ranked = pd.DataFrame(
            {
                "dataset": dataset,
                "model": [f"enc-{i:02d}" for i in range(N_RANKED)],
                "ranked": True,
                "croma": latent + rng.normal(scale=0.5, size=N_RANKED),
                "nipd_id": latent + rng.normal(scale=0.7, size=N_RANKED),
                "batch_asw": latent + rng.normal(scale=1.0, size=N_RANKED),
                "ilisi": latent + rng.normal(scale=1.5, size=N_RANKED),
            }
        )
        control = pd.DataFrame(
            {
                "dataset": [dataset],
                "model": [CONTROL],
                "ranked": [False],
                "croma": [0.0],
                "nipd_id": [0.0],
                "batch_asw": [0.0],
                "ilisi": [0.0],
            }
        )
        frames.extend([ranked, control])
    return pd.concat(frames, ignore_index=True)


def _row(table: pd.DataFrame, dataset: str, kind: str, metric: str) -> pd.Series:
    match = table[
        (table["dataset"] == dataset) & (table["kind"] == kind) & (table["metric"] == metric)
    ]
    assert len(match) == 1, f"expected one ({dataset}, {kind}, {metric}) row, got {len(match)}"
    return match.iloc[0]


def _infer(per_encoder: pd.DataFrame) -> pd.DataFrame:
    return study.associations(per_encoder, n_boot=300, n_permutations=500, seed=7)


def _two_by_two(centre_shift: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two biological classes x two centres, 60 samples per cell, in 8 dimensions.

    The classes sit apart along one axis; the centres are offset by ``centre_shift`` along
    another, so ``0`` is a perfectly mixed batch and a large shift a fully separated one.
    """
    rng = np.random.default_rng(seed)
    features, labels, batches = [], [], []
    for label in (0, 1):
        for centre in (0, 1):
            cell = rng.normal(scale=0.3, size=(60, 8))
            cell[:, 0] += 3.0 * label
            cell[:, 1] += centre_shift * centre
            cell[:, 2] += 2.0  # keep every row away from the origin before normalising
            features.append(cell)
            labels += [f"class-{label}"] * 60
            batches += [f"centre-{centre}"] * 60
    return np.vstack(features), np.array(labels), np.array(batches)


class TestScoring:
    """Per-encoder scores, against the reference package itself."""

    @pytest.fixture(autouse=True)
    def _scib(self) -> None:
        self.scib = pytest.importorskip("scib_metrics")

    def test_every_index_is_scored_on_normalised_and_raw_features(self) -> None:
        scores = study.score_encoder(*_two_by_two(0.0))
        assert set(scores) == {"batch_asw", "ilisi", "batch_asw_raw", "ilisi_raw"}
        assert all(np.isfinite(v) for v in scores.values())

    def test_mixed_centres_score_higher_than_separated_centres(self) -> None:
        mixed = study.score_encoder(*_two_by_two(0.0))
        separated = study.score_encoder(*_two_by_two(5.0))
        for index in ("batch_asw", "ilisi"):
            assert mixed[index] > separated[index]
        assert mixed["ilisi"] > 0.8 and separated["ilisi"] < 0.05

    def test_normalised_scores_ignore_feature_norms_and_raw_scores_do_not(self) -> None:
        features, labels, batches = _two_by_two(1.0)
        scale = np.random.default_rng(3).uniform(0.2, 5.0, size=(len(features), 1))
        plain = study.score_encoder(features, labels, batches)
        rescaled = study.score_encoder(features * scale, labels, batches)
        for index in ("batch_asw", "ilisi"):
            assert rescaled[index] == pytest.approx(plain[index], abs=1e-5)
            assert rescaled[f"{index}_raw"] != pytest.approx(plain[f"{index}_raw"], abs=1e-3)

    def test_batch_asw_is_scibs_rescaled_silhouette_within_each_class(self) -> None:
        features, labels, batches = _two_by_two(1.0)
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)
        expected = self.scib.silhouette_batch(unit, labels, batches, rescale=True)
        assert study.score_encoder(features, labels, batches)["batch_asw"] == pytest.approx(
            float(expected), abs=1e-6
        )

    def test_ilisi_reads_an_exact_90_neighbour_graph_at_perplexity_30(self) -> None:
        # The graph is built independently here, from scipy's full distance matrix: the
        # 90 nearest rows of each sample, itself included (scib masks the self edge).
        from scipy.spatial.distance import cdist
        from scib_metrics.nearest_neighbors import NeighborsResults

        features, labels, batches = _two_by_two(1.0)
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)
        distances = cdist(unit, unit)
        indices = np.argsort(distances, axis=1, kind="stable")[:, :90]
        graph = NeighborsResults(
            indices=indices, distances=np.take_along_axis(distances, indices, axis=1)
        )
        expected = self.scib.ilisi_knn(graph, batches, perplexity=30, scale=True)
        assert study.score_encoder(features, labels, batches)["ilisi"] == pytest.approx(
            float(expected), abs=1e-6
        )


def _run_metrics() -> pd.DataFrame:
    return pd.DataFrame({"model": ["B", "A", CONTROL], "croma": [0.2, -0.1, 0.05], "ri": [1, 2, 3]})


def _apd_join() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["camelyon"] * 3 + ["tolkach"] * 3,
            "model": ["A", "B", CONTROL] * 2,
            "nipd_id": [-0.3, -0.1, -0.2, 9.0, 9.0, 9.0],
            "croma": [-0.1, 0.2, 0.05, 0.0, 0.0, 0.0],
        }
    )


def _scores(models: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": models,
            "batch_asw": np.linspace(0.1, 0.3, len(models)),
            "ilisi": np.linspace(0.0, 0.2, len(models)),
            "batch_asw_raw": np.linspace(0.2, 0.4, len(models)),
            "ilisi_raw": np.linspace(0.1, 0.3, len(models)),
        }
    )


class TestCohortJoin:
    def test_the_roster_is_the_runs_and_each_value_comes_from_its_source(self) -> None:
        cohort = study.join_cohort(
            "camelyon", _scores(["A", CONTROL, "B"]), _run_metrics(), _apd_join()
        )
        assert cohort["model"].tolist() == ["B", "A", CONTROL]
        assert (cohort["dataset"] == "camelyon").all()
        assert cohort["ranked"].tolist() == [True, True, False]
        assert cohort["croma"].tolist() == [0.2, -0.1, 0.05]
        assert cohort["nipd_id"].tolist() == [-0.1, -0.3, -0.2]
        assert cohort.loc[cohort["model"] == "A", "ilisi_raw"].item() == pytest.approx(0.1)
        assert {"batch_asw", "ilisi", "batch_asw_raw", "ilisi_raw"} <= set(cohort.columns)

    def test_a_run_model_without_downstream_cells_raises(self) -> None:
        join = _apd_join()
        join = join[~((join["dataset"] == "camelyon") & (join["model"] == "B"))]
        with pytest.raises(ValueError, match="B"):
            study.join_cohort("camelyon", _scores(["A", "B", CONTROL]), _run_metrics(), join)

    def test_a_run_model_without_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="A"):
            study.join_cohort("camelyon", _scores(["B", CONTROL]), _run_metrics(), _apd_join())

    def test_a_run_without_the_control_raises(self) -> None:
        metrics = _run_metrics()
        metrics = metrics[metrics["model"] != CONTROL]
        with pytest.raises(ValueError, match=CONTROL):
            study.join_cohort("camelyon", _scores(["A", "B"]), metrics, _apd_join())


class TestAssociations:
    def test_downstream_rho_is_spearman_over_the_ranked_encoders(self) -> None:
        per_encoder = _per_encoder()
        table = _infer(per_encoder)
        for dataset in COHORTS:
            ranked = per_encoder[(per_encoder["dataset"] == dataset) & per_encoder["ranked"]]
            for metric in ("croma", "batch_asw", "ilisi"):
                row = _row(table, dataset, "downstream", metric)
                expected = spearmanr(ranked[metric], ranked["nipd_id"]).statistic
                assert row["estimate"] == pytest.approx(expected)
                assert row["target"] == "nipd_id"
                assert row["n"] == N_RANKED

    def test_agreement_is_each_baseline_against_median_croma(self) -> None:
        per_encoder = _per_encoder()
        table = _infer(per_encoder)
        ranked = per_encoder[(per_encoder["dataset"] == "tcga_4x4") & per_encoder["ranked"]]
        for metric in ("batch_asw", "ilisi"):
            row = _row(table, "tcga_4x4", "agreement", metric)
            assert row["target"] == "croma"
            assert row["estimate"] == pytest.approx(
                spearmanr(ranked[metric], ranked["croma"]).statistic
            )
            assert row["ci_lo"] <= row["estimate"] <= row["ci_hi"]

    def test_the_control_enters_no_correlation(self) -> None:
        per_encoder = _per_encoder()
        moved = per_encoder.copy()
        control = moved["model"] == CONTROL
        moved.loc[control, ["croma", "nipd_id", "batch_asw", "ilisi"]] = [9.0, -9.0, 9.0, -9.0]
        pd.testing.assert_frame_equal(_infer(per_encoder), _infer(moved))

    def test_q_values_adjust_exactly_the_downstream_associations(self) -> None:
        table = _infer(_per_encoder())
        downstream = table[table["kind"] == "downstream"]
        assert len(downstream) == 3 * len(COHORTS)
        np.testing.assert_allclose(
            downstream["q_value"], benjamini_hochberg(downstream["p_value"].to_numpy())
        )
        others = table[table["kind"] != "downstream"]
        assert others["p_value"].isna().all() and others["q_value"].isna().all()
        assert others["ci_lo"].notna().all() and others["ci_hi"].notna().all()

    def test_head_to_head_is_croma_minus_the_baseline_on_one_encoder_draw(self) -> None:
        per_encoder = _per_encoder()
        table = _infer(per_encoder)
        ranked = per_encoder[(per_encoder["dataset"] == "camelyon") & per_encoder["ranked"]]
        croma, nipd = ranked["croma"].to_numpy(), ranked["nipd_id"].to_numpy()
        draws = encoder_bootstrap_draws(N_RANKED, n_boot=300, seed=7)
        for metric in ("batch_asw", "ilisi"):
            baseline = ranked[metric].to_numpy()
            direct = np.array(
                [
                    spearmanr(croma[i], nipd[i]).statistic
                    - spearmanr(baseline[i], nipd[i]).statistic
                    for i in draws
                ]
            )
            row = _row(table, "camelyon", "head_to_head", metric)
            point = spearmanr(croma, nipd).statistic - spearmanr(baseline, nipd).statistic
            assert row["estimate"] == pytest.approx(point)
            lo, hi = np.nanpercentile(direct, [2.5, 97.5])
            assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((lo, hi))

    def test_encoders_on_the_ilisi_floor_tie_whatever_rounding_they_carry(self) -> None:
        # scib-metrics returns a single-centre median neighbourhood (scaled iLISI 0) as 0 or
        # a few float32 epsilons either side of it. Those encoders are tied on the floor;
        # an encoder clearly off it keeps its place.
        eps = float(np.finfo(np.float32).eps)
        per_encoder = _per_encoder()
        camelyon = (per_encoder["dataset"] == "camelyon") & per_encoder["ranked"]
        noisy = np.array([0.0, eps, 5 * eps, -eps, 2 * eps, 0.0, eps, 3 * eps])
        floored = np.zeros_like(noisy)
        rest = per_encoder.loc[camelyon, "ilisi"].to_numpy()[len(noisy) :]
        rest = 1e-4 + np.abs(rest)  # off the floor, however small
        with_noise = per_encoder.copy()
        with_noise.loc[camelyon, "ilisi"] = np.concatenate([noisy, rest])
        tied = per_encoder.copy()
        tied.loc[camelyon, "ilisi"] = np.concatenate([floored, rest])

        table = _infer(with_noise)

        pd.testing.assert_frame_equal(table, _infer(tied))
        ranked = tied[camelyon]
        assert _row(table, "camelyon", "downstream", "ilisi")["estimate"] == pytest.approx(
            spearmanr(ranked["ilisi"], ranked["nipd_id"]).statistic
        )
        assert study.at_ilisi_floor(noisy).all()
        assert not study.at_ilisi_floor(rest).any()

    def test_inference_is_deterministic(self) -> None:
        pd.testing.assert_frame_equal(_infer(_per_encoder()), _infer(_per_encoder()))


def test_without_the_reference_package_the_study_stops_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "scib_metrics", None)  # import raises ImportError

    def scored(*args, **kwargs):
        raise AssertionError("cells were scored without scib-metrics")

    monkeypatch.setattr(study, "_score_all", scored)
    with pytest.raises(SystemExit, match=r"croma\[repro\]"):
        study.main(["--workers", "1"])
