"""The MaRI-versus-RI study: its decomposition, its tau sweep and its shared-k leave-one-out.

Every expected value below is worked by hand from a six-anchor neighbourhood whose neighbour
types and distances are written out literally. The neighbour lists need not be geometrically
realisable, because scoring reads only each neighbour's label, confounder and distance.

    anchor  cell    top-3 neighbours (type, distance)               SO  OS
    0       L0 C0   s2 SO 0.10, s3 OS 0.20, s1 SS 0.30               1   1
    1       L0 C0   s0 SS 0.05, s4 OO 0.10, s5 OO 0.20               0   0   (unsupported)
    2       L0 C1   s0 SO 0.10, s1 SO 0.30, s4 OS 0.40               2   1
    3       L1 C0   s4 SO 0.20, s0 OS 0.20, (none: 2 valid)          1   1
    4       L1 C1   s5 SS 0.10, s3 SO 0.20, s2 OS 0.30               1   1
    5       L1 C1   s4 SS 0.10, s3 SO 0.25, s0 OO 0.50               1   0

So RI = 6 / 10 and support = 5 / 6.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import mari_vs_ri as study  # noqa: E402

K = 3
TAU = 0.1

E = np.exp
#: Per-anchor SO and OS weights at TAU, transcribed from the table in the module docstring.
SO_W = np.array([E(-1), 0.0, E(-1) + E(-3), E(-2), E(-2), E(-2.5)])
OS_W = np.array([E(-2), 0.0, E(-4), E(-2), E(-3), 0.0])
N_TYPED = np.array([2, 0, 3, 2, 2, 1])
RI = 6 / 10
MARI = SO_W.sum() / (SO_W.sum() + OS_W.sum())


def _hand_built() -> "study.Neighbours":
    return study.Neighbours(
        labels=np.array([0, 0, 0, 1, 1, 1]),
        centers=np.array([0, 0, 1, 0, 1, 1]),
        neigh_idx=np.array([[2, 3, 1], [0, 4, 5], [0, 1, 4], [4, 0, -1], [5, 3, 2], [4, 3, 0]]),
        neigh_dist=np.array(
            [
                [0.10, 0.20, 0.30],
                [0.05, 0.10, 0.20],
                [0.10, 0.30, 0.40],
                [0.20, 0.20, np.inf],
                [0.10, 0.20, 0.30],
                [0.10, 0.25, 0.50],
            ]
        ),
        valid_counts=np.array([3, 3, 3, 2, 3, 3]),
    )


def test_decomposition_reproduces_pooled_ri_mari_and_support() -> None:
    row = study.decompose(_hand_built(), k=K, tau=TAU)

    assert row["ri"] == pytest.approx(RI)
    assert row["mari"] == pytest.approx(MARI)
    assert row["support"] == pytest.approx(5 / 6)
    assert row["delta"] == pytest.approx(MARI - RI)


def test_logit_gap_is_the_log_ratio_of_mean_typed_weights() -> None:
    row = study.decompose(_hand_built(), k=K, tau=TAU)

    # Six SO slots and four OS slots in all, pooled across anchors.
    assert row["w_so"] == pytest.approx(SO_W.sum() / 6)
    assert row["w_os"] == pytest.approx(OS_W.sum() / 4)
    expected_gap = np.log(MARI / (1 - MARI)) - np.log(RI / (1 - RI))
    assert row["logit_gap"] == pytest.approx(expected_gap)
    assert row["log_w_ratio"] == pytest.approx(expected_gap)


def test_weight_ratio_is_the_median_nearest_to_farthest_weight_over_anchors() -> None:
    """Per anchor, exp((d_last - d_1) / tau) over its valid top-k neighbours of any type.

    The six anchors give e^2, e^1.5, e^3, 1 (anchor 3: two valid neighbours, tied), e^2 and
    e^4, whose median is e^2.
    """
    row = study.decompose(_hand_built(), k=K, tau=TAU)

    assert row["weight_ratio"] == pytest.approx(E(2))


def test_delta_splits_into_within_and_between_anchor_parts() -> None:
    """Within: each anchor's weighted SO share, count-weighted, minus RI. Between: the rest."""
    supported = N_TYPED > 0
    r_w = SO_W[supported] / (SO_W[supported] + OS_W[supported])
    count_weighted = (N_TYPED[supported] * r_w).sum() / N_TYPED.sum()

    row = study.decompose(_hand_built(), k=K, tau=TAU)

    assert row["delta_within"] == pytest.approx(count_weighted - RI)
    assert row["delta_between"] == pytest.approx(MARI - count_weighted)
    assert row["delta_within"] + row["delta_between"] == pytest.approx(row["delta"])


def test_effective_anchor_share_is_kish_size_over_all_anchors() -> None:
    """Typed-slot counts 2, 0, 3, 2, 2, 1: Kish size 10^2 / 22 over six anchors."""
    row = study.decompose(_hand_built(), k=K, tau=TAU)

    assert row["effective_anchor_share"] == pytest.approx((100 / 22) / 6)


def test_tau_sweep_rescores_the_same_neighbours_at_each_scaled_tau() -> None:
    """At f = 1/2 (tau = 0.05) every exponent doubles; at f = 1 the sweep is MaRI itself."""
    so_half = np.array([E(-2), 0.0, E(-2) + E(-6), E(-4), E(-4), E(-5)])
    os_half = np.array([E(-4), 0.0, E(-8), E(-4), E(-6), 0.0])
    anchor_weight = so_half + os_half

    rows = {r["f"]: r for r in study.tau_sweep(_hand_built(), k=K, tau=TAU, factors=(0.5, 1.0))}

    half = rows[0.5]
    assert half["tau_f"] == pytest.approx(TAU / 2)
    assert half["mari"] == pytest.approx(so_half.sum() / anchor_weight.sum())
    assert half["delta"] == pytest.approx(half["mari"] - RI)
    kish = anchor_weight.sum() ** 2 / np.square(anchor_weight).sum()
    assert half["effective_anchor_share"] == pytest.approx(kish / 6)
    assert rows[1.0]["mari"] == pytest.approx(MARI)


def _published(**overrides: float) -> dict[str, float]:
    row = study.decompose(_hand_built(), k=K, tau=TAU)
    return {**{key: row[key] for key in ("k", "tau", "support", "ri", "mari")}, **overrides}


def test_reproduction_check_accepts_the_published_values_exactly() -> None:
    row = study.decompose(_hand_built(), k=K, tau=TAU)

    study.verify_reproduction(row, _published(), recomputed_tau=TAU, context="bench/model")


@pytest.mark.parametrize("column", ["ri", "mari", "support", "tau"])
def test_reproduction_check_rejects_a_one_ulp_departure(column: str) -> None:
    row = study.decompose(_hand_built(), k=K, tau=TAU)
    published = _published()
    published[column] = float(np.nextafter(published[column], 2.0))

    with pytest.raises(study.ReproductionError, match=column):
        study.verify_reproduction(row, published, recomputed_tau=TAU, context="bench/model")


def test_reproduction_check_rejects_a_tau_that_is_not_the_automatic_one() -> None:
    row = study.decompose(_hand_built(), k=K, tau=TAU)

    with pytest.raises(study.ReproductionError, match="automatic"):
        study.verify_reproduction(
            row, _published(), recomputed_tau=TAU * 1.01, context="bench/model"
        )


def test_reproduction_check_rejects_a_broken_identity() -> None:
    row = {**study.decompose(_hand_built(), k=K, tau=TAU)}
    row["log_w_ratio"] += 1e-6

    with pytest.raises(study.ReproductionError, match="identity"):
        study.verify_reproduction(row, _published(), recomputed_tau=TAU, context="bench/model")


def test_leave_one_out_records_the_shared_k_after_each_single_removal() -> None:
    """k* 11, 11, 21, 31, 41: the lower median is 21; with four left it is the 2nd value.

    Removing either 11 leaves 11, 21, 31, 41 (shared k stays 21); removing 21, 31 or the
    control's 41 leaves two 11s at the bottom, so the shared k drops to 11.
    """
    k_star = {"A": 11, "B": 11, "C": 21, "D": 31, "Ctrl": 41}

    rows = {r["removed"]: r for r in study.leave_one_out(k_star)}

    assert {m: r["loo_k"] for m, r in rows.items()} == {
        "A": 21,
        "B": 21,
        "C": 11,
        "D": 11,
        "Ctrl": 11,
    }
    assert [m for m, r in rows.items() if r["moves_k"]] == ["C", "D", "Ctrl"]
    assert rows["D"]["k_star"] == 31
    assert all(r["run_k"] == 21 for r in rows.values())


def _sweep(rows: list[tuple[str, int, float, float, float]]) -> "pd.DataFrame":
    return pd.DataFrame(
        [
            {"model": m, "k": k, "knn_bacc": b, "ri": ri, "mari": mari, "k_values": "1,3,5"}
            for m, k, b, ri, mari in rows
        ]
    )


def test_k_star_is_the_first_best_balanced_accuracy_in_grid_order() -> None:
    """A ties at k = 3 and 5 and keeps the earlier; B's rows arrive out of grid order."""
    sweep = _sweep(
        [
            ("A", 1, 0.80, 0, 0),
            ("A", 3, 0.90, 0, 0),
            ("A", 5, 0.90, 0, 0),
            ("B", 5, 0.95, 0, 0),
            ("B", 1, 0.70, 0, 0),
            ("B", 3, 0.80, 0, 0),
        ]
    )

    assert study.k_star_from_sweep(sweep) == {"A": 3, "B": 5}


def test_alternative_k_effect_compares_ranked_orderings_between_the_two_ks() -> None:
    """At k = 3 -> 1, C and D swap (one discordant pair); ranks 1,2,3,4 vs 1,2,4,3 give
    Spearman 1 - 6*2/(4*15) = 0.8. The control moves most and must not count."""
    sweep = _sweep(
        [
            ("A", 3, 0, 0.50, 0.50),
            ("B", 3, 0, 0.60, 0.60),
            ("C", 3, 0, 0.70, 0.70),
            ("D", 3, 0, 0.80, 0.80),
            ("Ctrl", 3, 0, 0.10, 0.10),
            ("A", 1, 0, 0.52, 0.51),
            ("B", 1, 0, 0.58, 0.60),
            ("C", 1, 0, 0.75, 0.70),
            ("D", 1, 0, 0.74, 0.90),
            ("Ctrl", 1, 0, 0.90, 0.90),
        ]
    )

    effect = study.alternative_k_effect(sweep, run_k=3, alt_k=1, models=["A", "B", "C", "D"])

    assert effect["max_abs_delta_ri"] == pytest.approx(0.06)
    assert effect["max_abs_delta_mari"] == pytest.approx(0.10)
    assert effect["swapped_pairs"] == 1
    assert effect["rho_ri"] == pytest.approx(0.8)


def _panel_sweep() -> "pd.DataFrame":
    """k* A=1, B=1, C=3, D=5, Ctrl=5: shared k 3; dropping C, D or Ctrl moves it to 1.

    At k = 1, B and C swap places in RI and nothing else moves.
    """
    bacc_peak = {"A": 1, "B": 1, "C": 3, "D": 5, "Ctrl": 5}
    ri = {
        1: {"A": 0.50, "B": 0.65, "C": 0.60, "D": 0.80, "Ctrl": 0.2},
        3: {"A": 0.50, "B": 0.60, "C": 0.70, "D": 0.80, "Ctrl": 0.4},
        5: {"A": 0.50, "B": 0.60, "C": 0.70, "D": 0.80, "Ctrl": 0.4},
    }
    rows = []
    for model, peak in bacc_peak.items():
        for k in (1, 3, 5):
            bacc = 0.9 if k == peak else 0.8
            rows.append((model, k, bacc, ri[k][model], ri[k][model]))
    return _sweep(rows)


def test_shared_k_loo_reports_the_effect_of_each_removal_that_moves_k() -> None:
    table = study.shared_k_loo(_panel_sweep(), run_k=3, ranked=list("ABCD")).set_index("removed")

    assert table["moves_k"].sum() == 3
    moved = table[table["moves_k"]]
    assert set(moved["loo_k"]) == {1}
    assert moved["max_abs_delta_ri"].to_numpy() == pytest.approx([0.10] * 3)
    assert moved["max_abs_delta_mari"].to_numpy() == pytest.approx([0.10] * 3)
    assert set(moved["swapped_pairs"]) == {1}
    assert moved["rho_ri"].to_numpy() == pytest.approx([0.8] * 3)
    assert table.loc[["A", "B"], "max_abs_delta_ri"].isna().all()


def test_shared_k_loo_fails_when_the_rule_does_not_give_back_the_runs_k() -> None:
    with pytest.raises(study.ReproductionError, match="shared k"):
        study.shared_k_loo(_panel_sweep(), run_k=5, ranked=list("ABCD"))


def test_tau_sweep_summary_ranks_ranked_encoders_per_benchmark_and_factor() -> None:
    """At f = 1/2 MaRI ranks 1,4,3,2 against RI's 1,2,3,4: Spearman 1 - 6*8/60 = 0.2."""
    ri = [0.50, 0.62, 0.69, 0.85]
    by_f = {1.0: [0.50, 0.60, 0.70, 0.80], 0.5: [0.50, 0.80, 0.70, 0.60]}
    rows = []
    for f, maris in by_f.items():
        for model, r, m, nid, nood in zip(
            "ABCD", ri, maris, [-0.04, -0.03, -0.02, -0.01], [-0.01, -0.02, -0.03, -0.04]
        ):
            rows.append(dict(model=model, f=f, ri=r, mari=m, nipd_id=nid, nipd_ood=nood))
        rows.append(dict(model="Ctrl", f=f, ri=0.1, mari=0.9, nipd_id=0.0, nipd_ood=0.0))
    sweep = pd.DataFrame(rows).assign(benchmark="bench", delta=lambda d: d.mari - d.ri)

    summary = study.summarise_tau_sweep(sweep, ranked=list("ABCD")).set_index("f")

    assert summary.loc[1.0, "n"] == 4
    assert summary.loc[1.0, "max_abs_delta"] == pytest.approx(0.05)
    assert summary.loc[0.5, "max_abs_delta"] == pytest.approx(0.25)
    assert summary.loc[1.0, "rho_ri"] == pytest.approx(1.0)
    assert summary.loc[0.5, "rho_ri"] == pytest.approx(0.2)
    assert summary.loc[0.5, "rho_nipd_id"] == pytest.approx(0.2)
    assert summary.loc[0.5, "rho_nipd_ood"] == pytest.approx(-0.2)
