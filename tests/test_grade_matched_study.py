"""The grade-matched SO-pool study on PCaBiop (``scripts/studies/grade_matched_pools.py``).

The study asks whether a cancer anchor's negative CRoMa comes from its same-label,
other-centre (SO) neighbours being a *different ISUP grade*. It restricts the SO pool to
the anchor's grade and compares the margin against a random SO pool of the same size. The
tests drive the study's pieces on toy cohorts: the grade join, the sub-cohort that
restricts the SO pool, the seeded random pools, the full-pool exactness check, and the
measures read off the three pools. None of them needs ``output/`` or ``data/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import grade_matched_pools as gmp  # noqa: E402

PROVIDERS = ("karolinska", "radboud")


def _toy_manifests(per_grade: int = 2, n_benign: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A PCaBiop-shaped cohort and its ISUP manifest: benign "0", cancer "1" of grades 1-5."""
    rows, isup = [], []
    for provider in PROVIDERS:
        for i in range(n_benign):
            sid = f"{provider}-benign-{i}"
            rows.append({"sample_id": sid, "label": "0", "confounder": provider})
            isup.append({"sample_id": sid, "label": "0", "confounder": provider})
        for grade in range(1, 6):
            for i in range(per_grade):
                sid = f"{provider}-g{grade}-{i}"
                rows.append({"sample_id": sid, "label": "1", "confounder": provider})
                isup.append({"sample_id": sid, "label": str(grade), "confounder": provider})
    manifest = pd.DataFrame(rows)
    manifest["group_id"] = manifest["sample_id"]
    return manifest, pd.DataFrame(isup)


class TestCancerGrades:
    def test_every_cancer_slide_takes_its_grade_from_the_isup_manifest(self):
        manifest, isup = _toy_manifests(per_grade=2)

        grades = gmp.cancer_grades(manifest, isup, per_grade=2)

        cancer = manifest[manifest["label"] == "1"]
        assert list(grades.index) == list(cancer.index)
        assert grades.loc[cancer.index[0]] == 1
        assert grades.to_dict() == {
            idx: int(sid.split("-g")[1][0]) for idx, sid in cancer["sample_id"].items()
        }

    def test_a_cancer_slide_without_a_grade_is_an_error(self):
        manifest, isup = _toy_manifests(per_grade=2)
        isup = isup[isup["sample_id"] != "radboud-g3-1"]

        with pytest.raises(ValueError, match="radboud-g3-1"):
            gmp.cancer_grades(manifest, isup, per_grade=2)

    def test_a_cancer_slide_graded_benign_is_an_error(self):
        manifest, isup = _toy_manifests(per_grade=2)
        isup.loc[isup["sample_id"] == "karolinska-g2-0", "label"] = "0"

        with pytest.raises(ValueError, match="karolinska-g2-0"):
            gmp.cancer_grades(manifest, isup, per_grade=2)

    def test_a_cohort_off_the_stated_grade_balance_is_an_error(self):
        # Regrading one slide leaves every slide graded but radboud with 3 grade-4 and
        # 1 grade-5 slides: the balance is what the study claims, so it is checked.
        manifest, isup = _toy_manifests(per_grade=2)
        isup.loc[isup["sample_id"] == "radboud-g5-0", "label"] = "4"

        with pytest.raises(ValueError, match="radboud"):
            gmp.cancer_grades(manifest, isup, per_grade=2)

    def test_the_balance_is_checked_against_the_stated_count(self):
        manifest, isup = _toy_manifests(per_grade=2)

        with pytest.raises(ValueError, match="3"):
            gmp.cancer_grades(manifest, isup, per_grade=3)


def _typed(cohort: pd.DataFrame, anchor: pd.Series) -> tuple[set[int], set[int]]:
    """An anchor's SO and OS candidate rows within ``cohort``, as CRoMa types them."""
    same_label = cohort["label"] == anchor["label"]
    same_centre = cohort["confounder"] == anchor["confounder"]
    return set(cohort.index[same_label & ~same_centre]), set(
        cohort.index[~same_label & same_centre]
    )


def _rows(manifest: pd.DataFrame, provider: str, label: str) -> pd.Index:
    return manifest.index[(manifest["confounder"] == provider) & (manifest["label"] == label)]


class TestSubCohort:
    def test_an_anchors_so_set_is_the_chosen_pool_and_its_os_set_is_unchanged(self):
        manifest, _ = _toy_manifests(per_grade=2)
        pool = _rows(manifest, "radboud", "1")[[0, 3, 7]]

        rows = gmp.sub_cohort(manifest, anchor_provider="karolinska", pool=pool)

        cohort = manifest.iloc[rows]
        for _, anchor in manifest.loc[_rows(manifest, "karolinska", "1")].iterrows():
            so, os_ = _typed(cohort, anchor)
            assert so == set(pool)
            assert os_ == _typed(manifest, anchor)[1]

    def test_every_pooled_slide_keeps_both_typed_sets(self):
        # The library scores every row it is given, so the pooled slides need their own
        # SO (the anchor provider's cancers) and OS (their own provider's benigns).
        manifest, _ = _toy_manifests(per_grade=2)
        pool = _rows(manifest, "radboud", "1")[[1, 2]]

        cohort = manifest.iloc[gmp.sub_cohort(manifest, anchor_provider="karolinska", pool=pool)]

        for _, slide in manifest.loc[pool].iterrows():
            assert _typed(cohort, slide) == _typed(manifest, slide)

    def test_the_full_pool_is_the_whole_cohort(self):
        manifest, _ = _toy_manifests(per_grade=2)
        pool = _rows(manifest, "karolinska", "1")

        rows = gmp.sub_cohort(manifest, anchor_provider="radboud", pool=pool)

        assert list(rows) == list(range(len(manifest)))

    def test_a_pool_reaching_outside_the_other_providers_cancers_is_an_error(self):
        manifest, _ = _toy_manifests(per_grade=2)
        pool = list(_rows(manifest, "radboud", "1")[:2]) + [_rows(manifest, "radboud", "0")[0]]

        with pytest.raises(ValueError, match="pool"):
            gmp.sub_cohort(manifest, anchor_provider="karolinska", pool=pool)


#: One cancer anchor at angle 0 with its benign OS neighbour at 0.3 rad, and three cancer
#: SO candidates at the other provider, the nearest at 0.1 rad. Cosine distance on the unit
#: circle is 1 - cos(angle), so every margin below is read straight off the angles.
_ANGLES = {
    "anchor": ("karolinska", "1", 0.0),
    "os": ("karolinska", "0", 0.3),
    "near-so": ("radboud", "1", 0.1),
    "mid-so": ("radboud", "1", 0.5),
    "far-so": ("radboud", "1", 0.9),
    "other-benign": ("radboud", "0", 2.0),
}
_SEARCH_M1 = gmp.Search(m_values=(1,), headline_m=1, start_k=2, k_growth_factor=2.0, alpha=0.1)


def _circle_cohort() -> tuple[np.ndarray, pd.DataFrame]:
    manifest = pd.DataFrame(
        [
            {
                "sample_id": sid,
                "image_path": f"{sid}.tiff",
                "confounder": provider,
                "label": label,
                "group_id": sid,
            }
            for sid, (provider, label, _) in _ANGLES.items()
        ]
    )
    angles = np.array([angle for _, _, angle in _ANGLES.values()])
    return np.column_stack([np.cos(angles), np.sin(angles)]), manifest


def _margin(d_os: float, d_so: float) -> float:
    return (d_os - d_so) / (d_os + d_so)


class TestAnchorCroma:
    def test_the_margin_is_read_against_the_pool_alone(self):
        features, manifest = _circle_cohort()
        anchor = [0]
        pool = [3, 4]  # mid-so and far-so; the nearer near-so is left out

        rows = gmp.sub_cohort(manifest, anchor_provider="karolinska", pool=pool)
        croma = gmp.anchor_croma(features, manifest, rows=rows, anchors=anchor, search=_SEARCH_M1)

        expected = _margin(1 - np.cos(0.3), 1 - np.cos(0.5))
        assert croma == pytest.approx([expected], abs=1e-12)

    def test_the_full_pool_reads_the_nearest_so_neighbour(self):
        features, manifest = _circle_cohort()

        rows = gmp.sub_cohort(manifest, anchor_provider="karolinska", pool=[2, 3, 4])
        croma = gmp.anchor_croma(features, manifest, rows=rows, anchors=[0], search=_SEARCH_M1)

        assert croma == pytest.approx([_margin(1 - np.cos(0.3), 1 - np.cos(0.1))], abs=1e-12)

    def test_an_anchor_outside_the_sub_cohort_is_an_error(self):
        features, manifest = _circle_cohort()
        rows = gmp.sub_cohort(manifest, anchor_provider="karolinska", pool=[3, 4])

        with pytest.raises(ValueError, match="anchor"):
            gmp.anchor_croma(features, manifest, rows=rows, anchors=[2], search=_SEARCH_M1)


#: SO candidates per anchor provider: each provider's anchors draw from the other's cancers.
_CANDIDATES = {"karolinska": np.arange(100, 110), "radboud": np.arange(200, 210)}
_GROUPS = [(provider, grade) for provider in PROVIDERS for grade in (1, 2)]


class TestRandomPools:
    def test_one_pool_per_replicate_and_group_from_that_groups_candidates(self):
        pools = gmp.draw_random_pools(_CANDIDATES, _GROUPS, size=4, n_replicates=3, seed=0)

        assert pools.shape == (3, len(_GROUPS), 4)
        for r in range(3):
            for g, (provider, _) in enumerate(_GROUPS):
                assert len(set(pools[r, g])) == 4
                assert set(pools[r, g]) <= set(_CANDIDATES[provider])

    def test_the_seed_fixes_every_pool(self):
        first = gmp.draw_random_pools(_CANDIDATES, _GROUPS, size=4, n_replicates=5, seed=7)
        again = gmp.draw_random_pools(_CANDIDATES, _GROUPS, size=4, n_replicates=5, seed=7)
        other = gmp.draw_random_pools(_CANDIDATES, _GROUPS, size=4, n_replicates=5, seed=8)

        assert np.array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_draws_range_over_every_candidate_not_one_stratum(self):
        # A grade-stratified draw would confine each group's pools to a fifth of the
        # candidates; a uniform draw reaches all of them across replicates.
        pools = gmp.draw_random_pools(_CANDIDATES, _GROUPS, size=4, n_replicates=50, seed=0)

        for g, (provider, _) in enumerate(_GROUPS):
            assert set(pools[:, g].ravel()) == set(_CANDIDATES[provider])


class TestFullPoolCheck:
    def test_bit_identical_margins_pass(self):
        run = np.array([-0.42, 0.1, 0.3])

        gmp.require_full_pool_reproduces_run("TITAN", run=run, full_pool=run.copy())

    def test_a_one_ulp_difference_fails(self):
        # Exact means exact: a tolerance would let a wrong-but-close pool through.
        run = np.array([-0.42, 0.1, 0.3])
        full_pool = run.copy()
        full_pool[1] = np.nextafter(full_pool[1], 1.0)

        with pytest.raises(AssertionError, match="TITAN.*1/3"):
            gmp.require_full_pool_reproduces_run("TITAN", run=run, full_pool=full_pool)


#: Four anchors (two per grade) under four random replicates. Every expected value in
#: TestMeasures is worked out by hand from these literals.
_MARGINS = gmp.AnchorMargins(
    grade=np.array([1, 1, 2, 2]),
    full=np.array([-0.4, -0.2, 0.1, 0.3]),
    matched=np.array([-0.3, 0.1, 0.2, 0.4]),
    random=np.array(
        [
            [-0.5, -0.1, 0.0, 0.2],
            [-0.4, 0.0, 0.1, 0.3],
            [-0.3, 0.2, 0.2, 0.4],
            [-0.6, -0.2, 0.1, 0.1],
        ]
    ),
)


class TestMeasures:
    def _pooled(self) -> pd.Series:
        summary = gmp.summarize(_MARGINS)
        return summary.set_index("grade").loc["all"]

    def test_pool_summaries(self):
        s = self._pooled()

        assert s["n_anchors"] == 4
        assert s["full_median"] == pytest.approx(-0.05)
        assert s["full_f0"] == pytest.approx(0.5)
        assert s["matched_median"] == pytest.approx(0.15)
        assert s["matched_f0"] == pytest.approx(0.25)
        # Replicate medians -0.05, 0.05, 0.20, -0.05; replicate F(0)s 0.75, 0.5, 0.25, 0.5
        # (closed boundary: the 0.0 in replicate 0 is confounder-dominant).
        assert s["random_median"] == pytest.approx(0.0)
        assert s["random_f0"] == pytest.approx(0.5)

    def test_the_random_range_is_the_central_95_percent_of_replicate_medians(self):
        s = self._pooled()

        # Linear interpolation between order statistics -0.05, -0.05, 0.05, 0.20.
        assert s["random_median_lo"] == pytest.approx(-0.05)
        assert s["random_median_hi"] == pytest.approx(0.05 + 0.925 * 0.15)
        assert s["random_f0_lo"] == pytest.approx(0.25 + 0.075 * 0.25)
        assert s["random_f0_hi"] == pytest.approx(0.5 + 0.925 * 0.25)

    def test_grade_effect_and_its_one_sided_p(self):
        s = self._pooled()

        assert s["delta_grade"] == pytest.approx(0.15)
        assert s["p_one_sided"] == pytest.approx(0.25)  # only the 0.20 replicate reaches 0.15

    def test_per_anchor_paired_shift_and_sign_changes(self):
        s = self._pooled()

        # Replicate-mean random margins -0.45, -0.025, 0.1, 0.25; shifts 0.15, 0.125, 0.1, 0.15.
        assert s["paired_shift_median"] == pytest.approx(0.1375)
        # Two anchors are <= 0 under the random pool; matching lifts one of them above 0.
        assert s["n_random_nonpositive"] == 2
        assert s["n_sign_change"] == 1
        assert s["sign_change_share"] == pytest.approx(0.5)

    def test_each_grade_gets_its_own_grade_effect(self):
        by_grade = gmp.summarize(_MARGINS).set_index("grade")

        assert list(by_grade.index) == ["all", "1", "2"]
        # Grade 1: matched median -0.1; replicate medians -0.3, -0.2, -0.05, -0.4.
        assert by_grade.loc["1", "delta_grade"] == pytest.approx(-0.1 - (-0.25))
        assert by_grade.loc["1", "p_one_sided"] == pytest.approx(0.25)
        # Grade 2: matched median 0.3; replicate medians 0.1, 0.2, 0.3, 0.1.
        assert by_grade.loc["2", "delta_grade"] == pytest.approx(0.3 - 0.15)
        assert by_grade.loc["2", "p_one_sided"] == pytest.approx(0.25)


class TestEncoderMargins:
    """Each anchor is scored against its own group's pools, for all three pool kinds."""

    def _cohort(self):
        manifest, isup = _toy_manifests(per_grade=3, n_benign=3)
        manifest["image_path"] = manifest["sample_id"] + ".tiff"
        grades = gmp.cancer_grades(manifest, isup, per_grade=3)
        rng = np.random.default_rng(3)
        angles = rng.uniform(0.0, np.pi / 2, size=len(manifest))
        features = np.column_stack([np.cos(angles), np.sin(angles)])
        return manifest, grades, angles, features

    @staticmethod
    def _expected(manifest, angles, anchor: int, pool) -> float:
        """CRoMa(m=1) from its definition: nearest pooled SO slide against nearest OS slide."""
        row = manifest.iloc[anchor]
        os_rows = np.flatnonzero(
            (manifest["confounder"] == row["confounder"]).to_numpy()
            & (manifest["label"] != row["label"]).to_numpy()
        )
        d_so = min(1 - np.cos(angles[j] - angles[anchor]) for j in pool)
        d_os = min(1 - np.cos(angles[j] - angles[anchor]) for j in os_rows)
        return _margin(d_os, d_so)

    def test_full_matched_and_random_margins_use_each_anchors_own_pools(self):
        manifest, grades, angles, features = self._cohort()
        groups = [(p, g) for p in PROVIDERS for g in range(1, 6)]
        cancer_at = {
            p: np.flatnonzero(
                (manifest["confounder"] != p).to_numpy() & (manifest["label"] == "1").to_numpy()
            )
            for p in PROVIDERS
        }
        pools = gmp.draw_random_pools(cancer_at, groups, size=3, n_replicates=2, seed=0)

        margins = gmp.encoder_margins(
            features, manifest, grades, random_pools=pools, groups=groups, search=_SEARCH_M1
        )

        assert list(margins.grade) == list(grades)
        for i, (anchor, grade) in enumerate(grades.items()):
            provider = manifest.loc[anchor, "confounder"]
            same_grade = [j for j in cancer_at[provider] if grades[j] == grade]
            group = groups.index((provider, grade))
            assert margins.full[i] == pytest.approx(
                self._expected(manifest, angles, anchor, cancer_at[provider]), abs=1e-12
            )
            assert margins.matched[i] == pytest.approx(
                self._expected(manifest, angles, anchor, same_grade), abs=1e-12
            )
            for r in range(2):
                assert margins.random[r, i] == pytest.approx(
                    self._expected(manifest, angles, anchor, pools[r, group]), abs=1e-12
                )

    def test_splitting_a_groups_pools_across_tasks_changes_nothing(self):
        # The pools of one group are spread over worker tasks for load balance; the
        # reassembled margins must land on the same anchors and replicates.
        manifest, grades, _, features = self._cohort()
        groups = [(p, g) for p in PROVIDERS for g in range(1, 6)]
        cancer_at = {
            p: np.flatnonzero(
                (manifest["confounder"] != p).to_numpy() & (manifest["label"] == "1").to_numpy()
            )
            for p in PROVIDERS
        }
        pools = gmp.draw_random_pools(cancer_at, groups, size=3, n_replicates=4, seed=1)
        kwargs = dict(random_pools=pools, groups=groups, search=_SEARCH_M1)

        whole = gmp.encoder_margins(features, manifest, grades, **kwargs)
        split = gmp.encoder_margins(features, manifest, grades, pools_per_task=2, **kwargs)

        assert np.array_equal(whole.matched, split.matched)
        assert np.array_equal(whole.random, split.random)
