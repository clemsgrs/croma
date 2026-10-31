"""Inference for the downstream-correlation table: CIs, permutation p, BH q, partial rows.

The expectations are worked by hand: five perfectly concordant encoders give rho = 1, and
of the 5! = 120 orderings exactly two (the identity and the reversal) reach |rho| = 1, so
the exact two-sided p is 2 / 120.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "scripts" / "studies" / "apd"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

import apd_croma_correlation as study  # noqa: E402
from croma.metrics.bootstrap import encoder_bootstrap_draws, spearman_inference  # noqa: E402

TARGETS = ("nipd_id", "nipd_ood", "apd_id", "apd_ood")
METRICS = ("croma", "ri", "mari", "croma_ltm_alpha", "croma_f0")


def _joined(scope: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"dataset": scope, "model": [f"{scope}-{i}" for i in range(n)]})
    for column in METRICS + TARGETS:
        frame[column] = rng.normal(size=n)
    return frame


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    pcabiop = _joined("pcabiop", 5, 0)
    pcabiop["croma"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    pcabiop["nipd_id"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    joined = pd.concat([_joined("camelyon", 10, 1), pcabiop], ignore_index=True)
    return study.association_table(joined, n_boot=50, seed=0, n_permutations=200)


def _row(table: pd.DataFrame, target: str, metric: str, scope: str) -> pd.Series:
    rows = table[(table.target == target) & (table.metric == metric) & (table.scope == scope)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_the_bh_family_is_every_rendered_tile_association(table):
    family = table[table["in_bh_family"]]
    assert set(family["scope"]) == {"camelyon"}
    expected = {("nipd_id", m) for m in ("croma", "ri", "mari", "croma_ltm_alpha", "croma_f0")} | {
        ("nipd_id", "croma_ltm_alpha_given_croma"),
        ("nipd_id", "croma_f0_given_croma"),
    }
    expected |= {("nipd_ood", m) for m in ("croma", "ri", "mari")}
    expected |= {("apd_id", "croma"), ("apd_ood", "croma")}
    assert set(zip(family["target"], family["metric"])) == expected
    assert family["q_bh"].notna().all()
    assert (family["q_bh"] >= family["perm_p"]).all()
    assert table.loc[~table["in_bh_family"], "q_bh"].isna().all()


def test_pcabiop_is_descriptive_with_an_exact_permutation_p(table):
    row = _row(table, "nipd_id", "croma", "pcabiop")
    assert row["spearman"] == 1.0
    assert row["permutation"] == "exact"
    assert row["n_permutations"] == 120
    assert row["perm_p"] == 2 / 120
    assert not row["in_bh_family"]


def test_tail_rows_are_plain_and_partial_on_the_median_in_domain_only(table):
    partial = table[table["partial_on"] == "croma"]
    assert set(partial["target"]) == {"nipd_id"}
    assert set(partial["metric"]) == {"croma_ltm_alpha_given_croma", "croma_f0_given_croma"}
    assert set(partial["scope"]) == {"camelyon"}
    plain = _row(table, "nipd_id", "croma_ltm_alpha", "camelyon")
    assert plain["partial_on"] == ""


def test_two_level_reduces_to_the_encoder_bootstrap_when_runs_and_slides_do_not_vary():
    # Four encoders, three Cramer's-V splits, four identical probe runs each: resampling
    # the runs cannot move nIPD, and constant slide replicates cannot move the median.
    croma = np.array([0.1, 0.4, 0.2, 0.3])
    accuracies = [np.tile(np.array([[0.9], [0.8 + 0.02 * j], [0.7]]), (1, 4)) for j in range(4)]
    cramers_v = [0.0, 0.5, 1.0]
    draws = encoder_bootstrap_draws(4, n_boot=40, seed=7)
    reps = study.two_level_replicates(
        np.tile(croma, (40, 1)), accuracies, cramers_v=cramers_v, chance=0.5, draws=draws, seed=3
    )
    nipd = np.array([study.nipd(a, cramers_v=cramers_v, chance=0.5) for a in accuracies])
    expected = spearman_inference(croma, nipd, draws=draws).replicates
    np.testing.assert_array_equal(reps, expected)
