"""The sample-shortcut study: do low per-sample margins predict shortcut-induced errors?

``scripts/studies/sample_shortcut.py`` joins the probe records of the confounder-biased
sweep to a run's per-sample CRoMa, builds two baselines from the run's embeddings, and
scores each predictor by its within-cell AUROC for the *flipped* event. These tests drive
its public functions on hand-built toy cohorts whose answers can be worked out by hand.
They need no paper tree, so they stay in the source distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
for _p in (STUDIES, STUDIES / "apd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sample_shortcut as ss  # noqa: E402
import tile_colour  # noqa: E402

RED, BLUE, WHITE = (255, 0, 0), (0, 0, 255), (255, 255, 255)


def _tile(*colours, size: int = 8) -> np.ndarray:
    """A ``size`` x ``size`` RGB tile made of equal horizontal bands of ``colours``."""
    bands = np.array_split(np.arange(size), len(colours))
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    for rows, colour in zip(bands, colours):
        tile[rows] = colour
    return tile


class TestTileColour:
    def test_cielab_reproduces_reference_srgb_colours(self):
        lab = tile_colour.srgb_to_lab(np.array([[RED, BLUE, WHITE, (128, 128, 128)]]))[0]

        # Reference CIELAB (D65) values of the sRGB primaries, white and mid-grey.
        np.testing.assert_allclose(lab[0], [53.2408, 80.0925, 67.2032], atol=2e-3)
        np.testing.assert_allclose(lab[1], [32.2970, 79.1875, -107.8602], atol=2e-3)
        np.testing.assert_allclose(lab[2], [100.0, 0.0, 0.0], atol=1e-2)
        np.testing.assert_allclose(lab[3], [53.5850, 0.0, 0.0], atol=1e-2)

    def test_statistics_are_taken_over_the_tissue_and_skip_white_background(self):
        stats = tile_colour.colour_statistics(_tile(WHITE, RED))

        assert stats["tissue_fraction"] == pytest.approx(0.5)
        assert not stats["tissue_fallback"]
        # Optical density -log10((I + 1) / 256): 0 for a saturated channel, log10(256) for 0.
        assert stats["od_r_mean"] == pytest.approx(0.0)
        assert stats["od_g_mean"] == pytest.approx(2.40824, abs=1e-5)
        assert stats["od_b_sd"] == pytest.approx(0.0)
        assert stats["lab_l_mean"] == pytest.approx(53.2408, abs=2e-3)
        assert stats["lab_a_mean"] == pytest.approx(80.0925, abs=2e-3)
        assert stats["lab_l_sd"] == pytest.approx(0.0, abs=1e-9)

    def test_spread_is_the_population_standard_deviation_over_tissue_pixels(self):
        stats = tile_colour.colour_statistics(_tile(RED, BLUE))

        assert stats["tissue_fraction"] == pytest.approx(1.0)
        assert stats["lab_l_sd"] == pytest.approx((53.2408 - 32.2970) / 2, abs=2e-3)
        assert stats["od_r_sd"] == pytest.approx(2.40824 / 2, abs=1e-5)

    def test_a_tile_without_tissue_falls_back_to_every_pixel_and_says_so(self):
        stats = tile_colour.colour_statistics(_tile(WHITE))

        assert stats["tissue_fraction"] == 0.0
        assert stats["tissue_fallback"]
        assert stats["lab_l_mean"] == pytest.approx(100.0, abs=1e-2)

    def test_names_twelve_colour_features(self):
        assert len(tile_colour.FEATURES) == 12
        assert set(tile_colour.FEATURES) <= set(tile_colour.colour_statistics(_tile(RED)))


class TestCohortColourStats:
    def _images(self, tmp_path):
        from PIL import Image

        paths = []
        for i, colours in enumerate([(WHITE, RED), (RED, BLUE), (WHITE,)]):
            path = tmp_path / "tiles" / f"tile{i}.png"
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(_tile(*colours)).save(path)
            paths.append(str(path))
        return [f"s{i}" for i in range(3)], paths

    def test_one_row_per_tile_in_the_given_order(self, tmp_path):
        ids, paths = self._images(tmp_path)

        table = tile_colour.cohort_colour_stats(ids, paths, tmp_path / "colour.csv")

        assert table["sample_id"].tolist() == ids
        assert table["tissue_fraction"].tolist() == pytest.approx([0.5, 1.0, 0.0])
        assert table["tissue_fallback"].tolist() == [False, False, True]
        assert table.loc[1, "lab_l_sd"] == pytest.approx((53.2408 - 32.2970) / 2, abs=2e-3)
        assert table["image_sha256"].str.len().tolist() == [64, 64, 64]

    def test_a_second_call_reads_the_cache_without_opening_an_image(self, tmp_path):
        ids, paths = self._images(tmp_path)
        first = tile_colour.cohort_colour_stats(ids, paths, tmp_path / "colour.csv")
        for path in paths:
            Path(path).unlink()

        again = tile_colour.cohort_colour_stats(ids, paths, tmp_path / "colour.csv")

        assert again.equals(first)

    def test_a_different_tile_list_is_not_served_from_the_cache(self, tmp_path):
        ids, paths = self._images(tmp_path)
        tile_colour.cohort_colour_stats(ids, paths, tmp_path / "colour.csv")

        table = tile_colour.cohort_colour_stats(ids[:2], paths[:2], tmp_path / "colour.csv")

        assert table["sample_id"].tolist() == ids[:2]

    def test_a_cache_edited_after_it_was_written_is_recomputed(self, tmp_path):
        ids, paths = self._images(tmp_path)
        cache = tmp_path / "colour.csv"
        first = tile_colour.cohort_colour_stats(ids, paths, cache)
        cache.write_text(cache.read_text().replace("0.5", "0.25", 1))

        again = tile_colour.cohort_colour_stats(ids, paths, cache)

        assert again.equals(first)


def _slides(n_slides: int, per_slide: int, centres):
    slides = np.repeat(np.arange(n_slides), per_slide)
    return slides, np.asarray(centres)[slides % len(centres)]


class TestColourCentreClassifier:
    def test_colour_that_shifts_with_the_centre_is_recognised(self):
        rng = np.random.default_rng(0)
        slides, centres = _slides(20, 10, ["X", "Y", "Z"])
        shift = {"X": 0.0, "Y": 8.0, "Z": 16.0}
        features = (
            rng.normal(size=(slides.size, 12))
            + np.array([shift[c] for c in centres])[:, None] * np.r_[1.0, np.zeros(11)]
        )

        result = tile_colour.colour_centre_classifier(features, centres, slides, seed=0)

        assert result.balanced_accuracy > 0.9
        assert np.median(result.confidence) > 0.8
        assert result.confidence.shape == (slides.size,)

    def test_no_slide_is_scored_by_a_model_that_saw_it(self):
        rng = np.random.default_rng(1)
        slides, centres = _slides(20, 10, ["X", "Y"])
        # Every slide has its own colour, the centres none: twelve dimensions are enough to
        # memorise twenty slide signatures, so only slide-grouped folds keep this at chance.
        signature = rng.normal(size=(20, 12))[slides]
        features = signature + 0.05 * rng.normal(size=signature.shape)

        result = tile_colour.colour_centre_classifier(features, centres, slides, seed=0)

        assert all(len(set(result.folds[slides == s])) == 1 for s in range(20))
        assert result.balanced_accuracy < 0.75

    def test_is_deterministic_given_the_seed(self):
        rng = np.random.default_rng(2)
        slides, centres = _slides(20, 5, ["X", "Y"])
        features = rng.normal(size=(slides.size, 12))

        first = tile_colour.colour_centre_classifier(features, centres, slides, seed=0)
        again = tile_colour.colour_centre_classifier(features, centres, slides, seed=0)

        np.testing.assert_array_equal(first.confidence, again.confidence)


def _unit(degrees: float) -> list[float]:
    radians = np.deg2rad(degrees)
    return [float(np.cos(radians)), float(np.sin(radians))]


def _cosine_distance(a: float, b: float) -> float:
    """1 - cos of the angle between two unit vectors at ``a`` and ``b`` degrees."""
    return 1.0 - float(np.cos(np.deg2rad(a - b)))


#: A toy cohort on the unit circle, so every cosine distance is 1 - cos(angle gap).
#: (angle, class, centre, slide). Sample 0 has a same-slide twin at 1 degree that must be
#: ignored, and its nearest same-class neighbour from another slide shares its centre.
TOY = [
    (0.0, "a", "X", "s1"),
    (1.0, "a", "X", "s1"),
    (5.0, "a", "X", "s2"),
    (10.0, "a", "Y", "s3"),
    (30.0, "b", "X", "s4"),
    (60.0, "b", "Y", "s5"),
    (95.0, "b", "Y", "s6"),
]


def _toy():
    angles = np.array([row[0] for row in TOY])
    features = np.array([_unit(a) for a in angles]) * 3.0  # scale must not matter
    labels = np.array([row[1] for row in TOY])
    centres = np.array([row[2] for row in TOY])
    slides = np.array([row[3] for row in TOY])
    return angles, features, labels, centres, slides


class TestBiologyOnlyMargin:
    def test_m1_uses_the_nearest_same_and_different_class_neighbour_from_any_centre(self):
        _, features, labels, _, slides = _toy()

        margin = ss.biology_only_margin(features, labels, slides, m=1)

        # Sample 0: its 1-degree twin shares slide s1 and is skipped, so the nearest
        # same-class neighbour is the 5-degree tile -- from its own centre, which the
        # baseline does not care about. The nearest other-class tile sits at 30 degrees.
        d_same, d_diff = _cosine_distance(0, 5), _cosine_distance(0, 30)
        assert margin[0] == pytest.approx((d_diff - d_same) / (d_diff + d_same))
        assert margin[0] == pytest.approx(0.9447, abs=1e-4)

    def test_m2_averages_the_two_nearest_of_each_kind(self):
        _, features, labels, _, slides = _toy()

        margin = ss.biology_only_margin(features, labels, slides, m=2)

        # Sample 4 (class b at 30 degrees): the other b tiles sit 30 and 65 degrees away,
        # the nearest a tiles 20 (10-degree tile) and 25 (5-degree tile) degrees away.
        d_same = (_cosine_distance(30, 60) + _cosine_distance(30, 95)) / 2
        d_diff = (_cosine_distance(30, 10) + _cosine_distance(30, 5)) / 2
        assert margin[4] == pytest.approx((d_diff - d_same) / (d_diff + d_same))
        assert margin[4] < 0  # it sits nearer the other class than its own


def _record(margins, held_out, labels, classes_named, *, cramers_v=None, centres=None):
    """A probe record shaped like ``probe_records.read_record``'s, for the given margins."""
    margins = np.asarray(margins, dtype=float)
    n_splits = margins.shape[0]
    n_rows = len(labels)
    centres = ["X"] * n_rows if centres is None else centres
    return {
        "meta": {
            "model": "Toy",
            "biological_classes": list(classes_named),
            "n_splits": n_splits,
            "cramers_v": list(np.linspace(0, 1, n_splits)) if cramers_v is None else cramers_v,
        },
        "held_out": np.asarray(held_out),
        "classes": np.arange(len(classes_named)),
        "margins_in_domain": margins,
        "id_sample_id": np.array([f"t{i}" for i in range(n_rows)]),
        "id_image_path": np.array([f"/t{i}.png" for i in range(n_rows)]),
        "id_group_id": np.array([f"slide{i}" for i in range(n_rows)]),
        "id_label": np.array(labels),
        "id_confounder": np.array(centres),
    }


class TestSampleTargets:
    def test_binary_errors_margins_and_the_flipped_event(self):
        # 3 splits x 2 replicates; replicate 0 holds out rows 0 and 1, replicate 1 rows 1
        # and 2; row 3 is never held out. Margins point towards "tumor" (classes[1]).
        margins = np.zeros((3, 2, 2, 1))
        margins[0, 0, :, 0] = [-2.0, 1.0]  # V=0: both right
        margins[0, 1, :, 0] = [3.0, -0.5]  # V=0: row 1 right, row 2 wrong
        margins[2, 0, :, 0] = [1.0, -1.0]  # V=1: both wrong
        margins[2, 1, :, 0] = [2.0, -0.5]  # V=1: row 1 right, row 2 wrong
        record = _record(
            margins, [[0, 1], [1, 2]], ["normal", "tumor", "tumor", "normal"], ["normal", "tumor"]
        )

        targets = ss.sample_targets(record).set_index("row")

        assert list(targets.index) == [0, 1, 2]  # never-held-out row 3 is dropped
        assert targets.loc[0, ["n_held_out", "e0", "e1"]].tolist() == [1, 0.0, 1.0]
        assert targets.loc[0, ["margin0", "margin1", "margin_drop"]].tolist() == [2.0, -1.0, 3.0]
        # Wrong in one of its two V=1 replicates: e1 = 0.5 still counts as flipped.
        assert targets.loc[1, ["n_held_out", "e0", "e1"]].tolist() == [2, 0.0, 0.5]
        assert targets.loc[1, "margin_drop"] == pytest.approx((1 + 3) / 2 - (-1 + 2) / 2)
        # Already wrong at V=0, so whatever V=1 does it cannot be flipped.
        assert targets.loc[2, ["e0", "e1"]].tolist() == [1.0, 1.0]
        assert targets["flipped"].tolist() == [True, True, False]

    def test_multiclass_margin_is_the_true_class_minus_the_strongest_other(self):
        margins = np.zeros((2, 1, 2, 3))
        margins[0, 0] = [[0.1, 0.2, 0.9], [0.5, 0.1, 0.0]]
        margins[1, 0] = [[1.0, 0.2, 0.4], [0.3, 0.1, 0.0]]
        record = _record(margins, [[0, 1]], ["c", "a"], ["a", "b", "c"])

        targets = ss.sample_targets(record).set_index("row")

        assert targets.loc[0, "margin0"] == pytest.approx(0.9 - 0.2)
        assert targets.loc[0, "margin1"] == pytest.approx(0.4 - 1.0)
        assert targets.loc[0, ["e0", "e1", "flipped"]].tolist() == [0.0, 1.0, True]
        assert targets.loc[1, "margin0"] == pytest.approx(0.5 - 0.1)
        assert targets.loc[1, ["e0", "e1", "flipped"]].tolist() == [0.0, 0.0, False]

    def test_refuses_a_schedule_that_does_not_end_fully_confounded(self):
        record = _record(np.zeros((2, 1, 1, 1)), [[0]], ["a"], ["a", "b"], cramers_v=[0.0, 0.8])
        with pytest.raises(ValueError, match="V = 1"):
            ss.sample_targets(record)


class TestJoinTargets:
    def _record(self):
        margins = np.zeros((2, 2, 2, 1))
        margins[0] = [[[-1.0], [1.0]], [[1.0], [-1.0]]]  # V=0: all right
        margins[1] = [[[1.0], [1.0]], [[1.0], [-1.0]]]  # V=1: row 0 wrong
        return _record(
            margins,
            [[0, 1], [1, 2]],
            ["normal", "tumor", "normal", "tumor"],
            ["normal", "tumor"],
            centres=["X", "Y", "X", "Y"],
        )

    def _per_sample(self):
        import pandas as pd

        rows = pd.DataFrame(
            {
                "model": ["Toy"] * 4 + ["Other"] * 4,
                "sample_id": [f"t{i}" for i in (3, 1, 0, 2)] * 2,
                "group_id": [f"slide{i}" for i in (3, 1, 0, 2)] * 2,
                "label": [9, 9, 9, 9] * 2,  # a run may code classes differently: unused
                "croma_m5": [0.3, 0.1, -0.2, 0.0, 9.0, 9.0, 9.0, 9.0],
            }
        )
        return rows

    def test_matches_every_held_out_row_to_its_own_per_sample_croma(self):
        joined = ss.join_targets(self._record(), self._per_sample()).set_index("sample_id")

        assert sorted(joined.index) == ["t0", "t1", "t2"]  # t3 was never held out
        assert joined["croma_m5"].to_dict() == {"t0": -0.2, "t1": 0.1, "t2": 0.0}
        assert joined["flipped"].to_dict() == {"t0": True, "t1": False, "t2": False}
        # Cells come from the record's own class names and centres.
        assert joined["cell"].to_dict() == {
            "t0": "normal | X",
            "t1": "tumor | Y",
            "t2": "normal | X",
        }

    def test_counts_the_rows_no_replicate_held_out(self):
        assert ss.never_held_out(self._record()) == 1

    def test_refuses_a_held_out_row_without_per_sample_metrics(self):
        per_sample = self._per_sample()
        per_sample = per_sample[per_sample["sample_id"] != "t1"]
        with pytest.raises(ValueError, match="match no per-sample"):
            ss.join_targets(self._record(), per_sample)


def _toy_record():
    """The circle toy as a probe record: every row held out, row 3 flipped by the shortcut."""
    margins = np.zeros((2, 1, len(TOY), 1))
    signs = np.array([-1.0 if row[1] == "a" else 1.0 for row in TOY])  # "b" is classes[1]
    margins[0, 0, :, 0] = signs  # V=0: every row right
    margins[1, 0, :, 0] = signs * np.where(np.arange(len(TOY)) == 3, -1.0, 1.0)
    record = _record(
        margins,
        [list(range(len(TOY)))],
        [row[1] for row in TOY],
        ["a", "b"],
        centres=[row[2] for row in TOY],
    )
    record["id_group_id"] = np.array([row[3] for row in TOY])
    return record


def _toy_per_sample():
    import pandas as pd

    return pd.DataFrame(
        {
            "model": "Toy",
            "sample_id": [f"t{i}" for i in range(len(TOY))],
            "group_id": [row[3] for row in TOY],
            "croma_m5": np.linspace(-0.3, 0.3, len(TOY)),
        }
    )


class TestAnalyseEncoder:
    def test_attaches_both_baselines_to_the_rows_they_were_computed_for(self):
        _, features, _, _, _ = _toy()

        frame, summary = ss.analyse_encoder(
            _toy_record(), _toy_per_sample(), features, k=2, m=1, n_boot=10, seed=0
        )
        frame = frame.set_index("sample_id")

        # The hand-worked values of TestBiologyOnlyMargin and TestCentrePurity, now reached
        # through the record's rows and the run's per-sample rows.
        assert frame.loc["t3", "flipped"]
        assert frame.loc["t0", "bio_margin"] == pytest.approx(0.9447, abs=1e-4)
        assert frame.loc["t0", "purity"] == pytest.approx(0.5)
        assert frame.loc["t3", "purity"] == pytest.approx(0.0)
        assert frame.loc["t3", "croma_m5"] == pytest.approx(0.0)
        assert summary["n_flipped"] == 1 and not summary["evaluable"]

    def test_colour_confidence_is_attached_by_record_row(self):
        _, features, _, _, _ = _toy()
        confidence = np.linspace(0.1, 0.7, len(TOY))

        frame, summary = ss.analyse_encoder(
            _toy_record(),
            _toy_per_sample(),
            features,
            k=2,
            m=1,
            colour_confidence=confidence,
            n_boot=10,
            seed=0,
        )
        frame = frame.set_index("sample_id")

        assert frame.loc["t4", "colour_confidence"] == pytest.approx(0.5)
        assert frame.loc["t4", "colour_stratum"] == "b | X | q1"  # alone in its cell
        assert "auroc_colour" in summary


class TestAlignRows:
    def _manifest(self, order):
        import pandas as pd

        record = _toy_record()
        return pd.DataFrame(
            {
                "sample_id": record["id_sample_id"][order],
                "group_id": record["id_group_id"][order],
                "confounder": record["id_confounder"][order],
            }
        )

    def test_finds_each_record_row_in_a_differently_ordered_view(self):
        order = np.array([6, 2, 0, 5, 1, 3, 4])
        manifest = self._manifest(order)

        rows = ss.align_rows(_toy_record(), manifest)

        assert manifest["sample_id"].to_numpy()[rows].tolist() == [f"t{i}" for i in range(7)]

    def test_refuses_a_view_whose_centre_disagrees_with_the_record(self):
        manifest = self._manifest(np.arange(7))
        manifest.loc[2, "confounder"] = "Z"
        with pytest.raises(ValueError, match="centre"):
            ss.align_rows(_toy_record(), manifest)

    def test_refuses_a_view_missing_a_record_row(self):
        with pytest.raises(ValueError, match="absent"):
            ss.align_rows(_toy_record(), self._manifest(np.arange(6)))


class TestWithinCellAuroc:
    #: Cell A is event-rich and scores high, cell B event-poor and scores low, but inside
    #: each cell every event scores *below* every non-event. Cell C holds events only.
    SCORE = np.array([0.6, 0.7, 0.75, 0.8, 0.1, 0.2, 0.3, 0.35, 0.95, 0.05])
    EVENT = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    CELL = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"])

    def test_pairs_only_within_cells_where_pooling_would_reverse_the_answer(self):
        from sklearn.metrics import roc_auc_score

        within = ss.within_cell_auroc(self.SCORE, self.EVENT, self.CELL)

        assert within == 0.0
        # Pooled, the cell effect wins: the event-rich cell scores high.
        assert roc_auc_score(self.EVENT, self.SCORE) > 0.5

    def test_a_cell_without_both_kinds_contributes_nothing(self):
        keep = self.CELL != "C"
        assert ss.within_cell_auroc(self.SCORE, self.EVENT, self.CELL) == ss.within_cell_auroc(
            self.SCORE[keep], self.EVENT[keep], self.CELL[keep]
        )

    def test_counts_pairs_not_cells_and_ties_as_half(self):
        # Cell A: 1 event vs 2 non-events, the event ties one and beats the other (1.5/2).
        # Cell B: 1 event vs 1 non-event, the event loses (0/1). Pairs, not cells, weigh.
        score = np.array([0.5, 0.5, 0.4, 0.1, 0.9])
        event = np.array([1, 0, 0, 1, 0], dtype=bool)
        cell = np.array(["A", "A", "A", "B", "B"])

        assert ss.within_cell_auroc(score, event, cell) == pytest.approx(1.5 / 3)

    def test_is_undefined_when_no_cell_holds_both_kinds(self):
        assert np.isnan(
            ss.within_cell_auroc(np.array([0.1, 0.2]), np.array([True, False]), np.array([0, 1]))
        )

    def test_is_undefined_on_an_empty_population(self):
        empty = np.array([])
        assert np.isnan(ss.within_cell_auroc(empty, empty.astype(bool), empty.astype(str)))
        assert np.isnan(ss.within_cell_auroc([], [], []))

    def test_a_weighted_call_equals_the_rows_repeated_that_often(self):
        rng = np.random.default_rng(3)
        score = rng.integers(0, 6, 40).astype(float)  # coarse, so ties abound
        event = rng.random(40) < 0.4
        cell = rng.integers(0, 3, 40)
        weights = rng.integers(0, 4, 40)
        repeated = np.repeat(np.arange(40), weights)

        assert ss.WithinCellAuroc(score, event, cell)(weights) == pytest.approx(
            ss.within_cell_auroc(score[repeated], event[repeated], cell[repeated])
        )


def _clustered_toy(seed: int = 0, n_slides: int = 24, per_slide: int = 15):
    """Slides nested in two cells; within a cell, events score lower on a noisy margin."""
    rng = np.random.default_rng(seed)
    slides = np.repeat(np.arange(n_slides), per_slide)
    cells = slides % 2
    event = rng.random(slides.size) < 0.3
    margin = rng.normal(size=slides.size) - 0.8 * event + 0.5 * cells
    return margin, event, cells, slides


class TestBootstrapAurocs:
    def test_differences_are_paired_on_one_shared_draw(self):
        margin, event, cells, slides = _clustered_toy()

        result = ss.bootstrap_aurocs(
            {"croma": (-margin, cells), "rescaled": (-np.exp(margin), cells)},
            event=event,
            cells=cells,
            groups=slides,
            differences=[("croma", "rescaled")],
            n_boot=200,
            seed=0,
        )

        # The two scores rank every row identically, so on a shared draw every replicate
        # difference is exactly zero, although each AUROC is itself uncertain.
        croma = result.auroc["croma"]
        assert croma.lo < croma.point < croma.hi
        assert croma.point == pytest.approx(ss.within_cell_auroc(-margin, event, cells))
        difference = result.difference[("croma", "rescaled")]
        assert (difference.point, difference.lo, difference.hi) == (0.0, 0.0, 0.0)

    def test_is_deterministic_given_the_seed(self):
        margin, event, cells, slides = _clustered_toy()

        def run(seed):
            out = ss.bootstrap_aurocs(
                {"croma": (-margin, cells)},
                event=event,
                cells=cells,
                groups=slides,
                n_boot=100,
                seed=seed,
            )
            return out.auroc["croma"].lo, out.auroc["croma"].hi

        assert run(0) == run(0)
        assert run(0) != run(1)

    def test_a_predictor_may_pair_within_finer_strata(self):
        margin, event, cells, slides = _clustered_toy()
        finer = cells * 10 + (np.arange(slides.size) % 5)

        result = ss.bootstrap_aurocs(
            {"croma": (-margin, cells), "croma_given_quintile": (-margin, finer)},
            event=event,
            cells=cells,
            groups=slides,
            n_boot=50,
            seed=0,
        )

        assert result.auroc["croma_given_quintile"].point == pytest.approx(
            ss.within_cell_auroc(-margin, event, finer)
        )


class TestWithinCellSpearman:
    def test_correlates_inside_cells_where_pooling_would_reverse_the_sign(self):
        from scipy.stats import spearmanr

        x = np.array([1.0, 2, 3, 4, 11, 12, 13, 14])
        y = np.array([4.0, 3, 2, 1, 14, 13, 12, 11])
        cells = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        assert ss.within_cell_spearman(x, y, cells) == pytest.approx(-1.0)
        assert spearmanr(x, y).statistic > 0.5

    def test_weights_each_cell_by_its_size(self):
        x = np.array([1.0, 2, 3, 4, 1, 2, 3, 4, 5, 6])
        y = np.array([1.0, 2, 3, 4, 6, 5, 4, 3, 2, 1])
        cells = np.array([0] * 4 + [1] * 6)

        assert ss.within_cell_spearman(x, y, cells) == pytest.approx((4 * 1 + 6 * -1) / 10)

    def test_partial_given_a_covariate_matches_the_first_order_formula(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(4)
        z = rng.normal(size=60)
        x = z + rng.normal(size=60)
        y = z + 0.5 * x + rng.normal(size=60)
        r_xy, r_xz, r_yz = (spearmanr(a, b).statistic for a, b in ((x, y), (x, z), (y, z)))
        expected = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

        assert ss.within_cell_spearman(x, y, np.zeros(60), z=z) == pytest.approx(expected)


def _encoder_frame(n_flipped_per_cell: int = 6, seed: int = 0):
    """Held-out rows of one encoder: 2 cells x 6 slides x 10 tiles, CRoMa low when flipped."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    slides = np.repeat(np.arange(12), 10)
    cells = np.where(slides % 2 == 0, "a | X", "b | Y")
    flipped = np.zeros(slides.size, dtype=bool)
    for cell in np.unique(cells):
        flipped[rng.choice(np.flatnonzero(cells == cell), n_flipped_per_cell, replace=False)] = True
    croma = rng.uniform(0.1, 0.5, slides.size) - 0.5 * flipped  # every flip below every other
    frame = pd.DataFrame(
        {
            "group_id": slides.astype(str),
            "cell": cells,
            "e0": 0.0,
            "e1": np.where(flipped, 1.0, 0.0),
            "flipped": flipped,
            "croma_m5": croma,
            "bio_margin": rng.normal(size=slides.size),
            "purity": rng.uniform(size=slides.size),
            "margin0": rng.normal(size=slides.size) + 2,
        }
    )
    frame["margin_drop"] = frame["margin0"] - rng.normal(size=slides.size)
    return frame


class TestEncoderSummary:
    def test_scores_low_croma_as_evidence_of_a_flip(self):
        summary = ss.encoder_summary(_encoder_frame(), n_boot=50, seed=0)

        assert summary["n_flipped"] == 12
        assert summary["evaluable"]
        assert summary["auroc_croma"] == 1.0
        assert summary["auroc_croma_lo"] == 1.0
        assert 0.0 < summary["auroc_bio"] < 1.0
        assert summary["diff_croma_bio"] == pytest.approx(1.0 - summary["auroc_bio"])
        assert summary["diff_croma_purity_lo"] <= summary["diff_croma_purity"]
        assert np.isfinite(summary["rho_drop"]) and np.isfinite(summary["rho_drop_given_margin0"])

    def test_rows_the_balanced_probe_already_got_wrong_are_left_out(self):
        frame = _encoder_frame()
        hard = frame.iloc[:20].assign(e0=1.0, flipped=False, croma_m5=-5.0)
        with_hard = pd_concat([frame, hard])

        summary = ss.encoder_summary(with_hard, n_boot=20, seed=0)

        assert summary["n_held_out"] == len(frame) + 20
        assert summary["n_balanced_right"] == len(frame)
        assert summary["auroc_croma"] == 1.0  # the hard, lowest-CRoMa rows are not negatives

    def test_too_few_flipped_samples_leave_no_interval(self):
        summary = ss.encoder_summary(_encoder_frame(n_flipped_per_cell=4), n_boot=20, seed=0)

        assert summary["n_flipped"] == 8
        assert not summary["evaluable"]
        assert summary["auroc_croma"] == 1.0  # the point is kept, the table shows a dash
        assert np.isnan(summary["auroc_croma_lo"]) and np.isnan(summary["diff_croma_bio_hi"])

    def test_a_balanced_probe_wrong_on_every_row_gives_a_row_with_no_auroc(self):
        frame = _encoder_frame().assign(e0=0.5, flipped=False)
        frame["colour_confidence"] = np.random.default_rng(1).uniform(size=len(frame))
        frame["colour_stratum"] = frame["cell"]

        summary = ss.encoder_summary(frame, n_boot=20, seed=0)

        assert summary["n_held_out"] == len(frame)
        assert summary["n_balanced_right"] == 0 and summary["n_flipped"] == 0
        assert not summary["evaluable"]
        for name in ("croma", "bio", "purity", "colour", "croma_given_colour"):
            assert np.isnan(summary[f"auroc_{name}"])
        assert np.isnan(summary["diff_croma_bio"]) and np.isnan(summary["diff_croma_purity"])

    def test_colour_predictors_when_the_cohort_has_colour(self):
        frame = _encoder_frame()
        frame["colour_confidence"] = np.where(frame["flipped"], 0.9, 0.4)
        frame["colour_stratum"] = frame["cell"]

        summary = ss.encoder_summary(frame, n_boot=20, seed=0)

        assert summary["auroc_colour"] == 1.0
        assert summary["auroc_croma_given_colour"] == 1.0
        assert summary["rho_croma_colour"] < 0
        assert "auroc_colour" not in ss.encoder_summary(_encoder_frame(), n_boot=20, seed=0)


class TestColourStrata:
    def test_splits_each_cell_into_equal_colour_confidence_quintiles(self):
        confidence = np.r_[np.linspace(0.9, 0.0, 10), np.linspace(0.0, 0.45, 10)]
        cells = np.array(["a"] * 10 + ["b"] * 10)

        strata = ss.colour_strata(confidence, cells)

        assert strata[:10].tolist() == [
            f"a | q{q}" for q in (5, 5, 4, 4, 3, 3, 2, 2, 1, 1)
        ]  # quintiles are within the cell, whatever the other cell's confidences
        assert strata[10:].tolist() == [f"b | q{q}" for q in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5)]

    def test_ties_are_broken_by_row_order_so_strata_stay_equal_sized(self):
        strata = ss.colour_strata(np.full(10, 0.5), np.array(["a"] * 10))

        assert strata.tolist() == [f"a | q{q}" for q in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5)]


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


class TestCentrePurity:
    def test_is_the_own_centre_share_of_the_k_nearest_neighbours_of_any_class(self):
        _, features, _, centres, slides = _toy()

        purity = ss.centre_purity(features, centres, slides, k=2)

        # Sample 0 (centre X, 0 degrees): its 1-degree twin is on its slide, so its two
        # nearest are the 5-degree (X) and 10-degree (Y) tiles.
        assert purity[0] == pytest.approx(0.5)
        # Sample 3 (centre Y, 10 degrees): nearest are 5 (X) and 1 (X) degrees.
        assert purity[3] == pytest.approx(0.0)
        # Sample 6 (centre Y, 95 degrees): nearest are 60 (Y) and 30 (X) degrees.
        assert purity[6] == pytest.approx(0.5)
        # Sample 4 (centre X, 30 degrees): nearest are 10 (Y) and 5 (X); at k=3 also 1 (X).
        assert purity[4] == pytest.approx(0.5)
        assert ss.centre_purity(features, centres, slides, k=3)[4] == pytest.approx(2 / 3)
