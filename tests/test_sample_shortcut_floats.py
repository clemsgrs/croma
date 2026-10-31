"""The sample-shortcut floats and macros: rendered only from a study that is still current.

``scripts/repro/_sample_shortcut.py`` reads ``output/studies/sample-shortcut/<cohort>/`` and
refuses it unless every input the study recorded (records, per-sample metrics, embeddings,
manifests, colour cache) still has the sha256 it was computed from.
``generate_sample_shortcut_floats.py`` renders the tables, the montage float and the macros
from it. These tests build a miniature study under a temporary root. They import
``scripts/repro``, so they are excluded from the source distribution.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "repro", ROOT / "scripts" / "studies" / "apd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _sample_shortcut as basis  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402
from loaders import DATASETS  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

CONTROL = "DINOv2-B"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encoder_row(model, *, flipped, auroc, lo, hi, bio=0.6, purity=0.55, colour=True):
    lo = float("nan") if lo is None else lo
    hi = float("nan") if hi is None else hi
    row = {
        "cohort": "camelyon",
        "model": model,
        "k": 11,
        "n_held_out": 900,
        "n_balanced_right": 850,
        "n_flipped": flipped,
        "evaluable": flipped >= 10,
    }
    estimates = {
        "auroc_croma": (auroc, lo, hi),
        "auroc_bio": (bio, bio - 0.05, bio + 0.05),
        "auroc_purity": (purity, purity - 0.05, purity + 0.05),
        "diff_croma_bio": (auroc - bio, lo - bio, hi - bio),
        "diff_croma_purity": (auroc - purity, lo - purity, hi - purity),
    }
    if colour:
        estimates["auroc_colour"] = (0.52, 0.48, 0.56)
        estimates["auroc_croma_given_colour"] = (auroc - 0.01, lo - 0.01, hi - 0.01)
    for key, (point, low, high) in estimates.items():
        if flipped < 10:
            low = high = float("nan")
        row.update({key: point, f"{key}_lo": low, f"{key}_hi": high})
    row.update({"rho_drop": -0.2, "rho_drop_given_margin0": -0.1})
    if colour:
        row["rho_croma_colour"] = -0.05
    return row


#: (model, run median CRoMa, flipped, AUROC, lo, hi): A and B clear 0.5, C straddles it,
#: D has too few flips, and the control would lead every column.
PANEL = [
    ("Alpha", 0.30, 40, 0.80, 0.70, 0.90),
    ("Beta", 0.10, 25, 0.70, 0.55, 0.85),
    ("Gamma", 0.20, 15, 0.55, 0.45, 0.65),
    ("Delta", 0.05, 6, 0.90, None, None),
    (CONTROL, 0.40, 60, 0.95, 0.90, 0.99),
]


def _tile_images(root: Path) -> list[str]:
    """Twenty small solid-colour PNG tiles under ``root``."""
    from PIL import Image

    paths = []
    for i in range(20):
        path = root / "tiles" / f"s{i:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            Image.fromarray(np.full((16, 16, 3), 10 * i, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def _samples(models, images, cells=("normal | RUMC", "tumor | UMCU")):
    """Twenty held-out tiles per encoder; cell 1 holds more flips than cell 0."""
    rows = []
    for model in models:
        for i in range(20):
            cell = cells[0] if i < 8 else cells[1]
            rows.append(
                {
                    "model": model,
                    "row": i,
                    "sample_id": f"s{i:02d}",
                    "group_id": f"slide{i % 4}",
                    "image_path": images[i],
                    "cell": cell,
                    "e0": 0.0 if i != 19 else 1.0,
                    "flipped": i in (1, 9, 10, 11),
                    "croma_m5": (i * 7 % 20) / 20 - 0.5,
                    "colour_confidence": 0.5 + i / 100,
                }
            )
    return pd.DataFrame(rows)


def make_study(root: Path, cohort: str = "camelyon", panel=PANEL, colour: bool = True) -> Path:
    """A miniature, internally consistent study of ``cohort`` under ``root``."""
    entry = by_benchmark(DATASETS[cohort]["benchmark"])
    metrics = root / entry.metrics_rel
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": [p[0] for p in panel], "croma": [p[1] for p in panel], "k": 11}).to_csv(
        metrics, index=False
    )
    embedding = root / "output" / "embeddings" / "toy" / "Alpha.npy"
    embedding.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding, np.arange(6.0))

    study = root / basis.STUDY_REL / cohort
    study.mkdir(parents=True, exist_ok=True)
    encoders = pd.DataFrame(
        [
            _encoder_row(m, flipped=f, auroc=a, lo=lo, hi=hi, colour=colour)
            for m, _croma, f, a, lo, hi in panel
        ]
    ).assign(cohort=cohort)
    encoders.to_csv(study / "encoders.csv", index=False)
    _samples([p[0] for p in panel], _tile_images(root)).to_csv(
        study / "samples.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    inputs = {path.relative_to(root).as_posix(): _sha(path) for path in (metrics, embedding)}
    info = {
        "cohort": cohort,
        "benchmark": DATASETS[cohort]["benchmark"],
        "n_tiles": 1000,
        "n_held_out": 900,
        "n_never_held_out": 100,
        "n_cells": 2,
        "n_encoders": len(panel),
        "min_flipped": 10,
        "flip_error_rate": 0.5,
        "n_boot": 2000,
        "boot_seed": 0,
        "croma_m": 5,
        "outputs": {name: _sha(study / name) for name in ("encoders.csv", "samples.csv.gz")},
        "inputs": inputs,
    }
    if colour:
        info.update(
            colour_balanced_accuracy=0.93,
            colour_folds=5,
            colour_seed=0,
            tissue_od_threshold=0.3,
            n_tissue_fallback=0,
            colour_bins=5,
        )
    (study / "cohort.json").write_text(json.dumps(info))
    return study


class TestLoad:
    def test_reads_a_current_study_with_the_control_set_apart(self, tmp_path):
        make_study(tmp_path)

        result = basis.load("camelyon", tmp_path)

        assert result.ranked["model"].tolist() == ["Alpha", "Gamma", "Beta", "Delta"]
        assert result.control["model"].tolist() == [CONTROL]

    def test_refuses_a_study_whose_input_changed_since_it_ran(self, tmp_path):
        make_study(tmp_path)
        np.save(tmp_path / "output" / "embeddings" / "toy" / "Alpha.npy", np.zeros(6))

        with pytest.raises(CaptionClaimError, match="Alpha.npy"):
            basis.load("camelyon", tmp_path)

    def test_refuses_a_study_whose_input_is_gone(self, tmp_path):
        make_study(tmp_path)
        (tmp_path / "output" / "embeddings" / "toy" / "Alpha.npy").unlink()

        with pytest.raises(CaptionClaimError, match="missing"):
            basis.load("camelyon", tmp_path)

    def test_refuses_an_output_edited_after_the_study_wrote_it(self, tmp_path):
        study = make_study(tmp_path)
        frame = pd.read_csv(study / "encoders.csv")
        frame.loc[0, "auroc_croma"] = 0.99
        frame.to_csv(study / "encoders.csv", index=False)

        with pytest.raises(CaptionClaimError, match="encoders.csv"):
            basis.load("camelyon", tmp_path)

    def test_a_study_that_never_ran_fails_rather_than_skipping(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="sample_shortcut.py"):
            basis.load("camelyon", tmp_path)


class TestSummary:
    def test_counts_and_medians_are_over_evaluable_pathology_encoders_only(self, tmp_path):
        make_study(tmp_path)

        summary = basis.summary(basis.load("camelyon", tmp_path))

        assert summary["n_ranked"] == 4
        assert summary["n_evaluable"] == 3  # Delta has six flips; the control never counts
        assert summary["croma_auroc_median"] == pytest.approx(0.70)
        assert summary["n_ci_above_half"] == 2  # Alpha, Beta; Gamma straddles 0.5
        assert summary["n_ci_below_half"] == 0
        assert summary["n_above_bio"] == 1  # only Alpha's paired interval clears 0
        assert summary["n_below_bio"] == 0
        assert summary["n_never_held_out"] == 100

    def test_tile_cohorts_add_the_colour_summary(self, tmp_path):
        make_study(tmp_path)

        summary = basis.summary(basis.load("camelyon", tmp_path))

        assert summary["colour_balanced_accuracy"] == pytest.approx(0.93)
        assert summary["croma_given_colour_median"] == pytest.approx(0.69)
        assert summary["colour_auroc_median"] == pytest.approx(0.52)
        assert summary["rho_colour_median"] == pytest.approx(-0.05)

    def test_slide_cohorts_have_no_colour_summary(self, tmp_path):
        make_study(tmp_path, cohort="pcabiop", colour=False)

        summary = basis.summary(basis.load("pcabiop", tmp_path))

        assert "colour_balanced_accuracy" not in summary


class TestMontageChoice:
    def test_takes_the_lower_median_encoder_and_its_most_flipped_cell(self, tmp_path):
        make_study(tmp_path)

        choice = basis.montage_choice(basis.load("camelyon", tmp_path), n=3)

        # Evaluable pathology AUROCs are 0.55 (Gamma), 0.70 (Beta), 0.80 (Alpha).
        assert choice.model == "Beta"
        assert choice.cell == "tumor | UMCU"  # three of its four flips
        assert choice.n_flipped_in_cell == 3

    def test_shows_the_extreme_croma_tiles_the_balanced_probe_got_right(self, tmp_path):
        make_study(tmp_path)

        choice = basis.montage_choice(basis.load("camelyon", tmp_path), n=3)

        assert choice.low["sample_id"].tolist() == ["s09", "s12", "s15"]
        assert choice.high["sample_id"].tolist() == ["s11", "s14", "s17"]
        assert choice.low["flipped"].tolist() == [True, False, False]
        assert "s19" not in set(choice.low["sample_id"]) | set(choice.high["sample_id"])


def _generator():
    import generate_sample_shortcut_floats as generator

    return generator


def _float(tex: str, label: str) -> str:
    """The one float in ``tex`` that carries ``label``."""
    import re

    blocks = re.split(r"\\end\{(?:table|figure)\}", tex)
    floats = [block for block in blocks if rf"\label{{{label}}}" in block]
    assert len(floats) == 1, f"{label}: {len(floats)} floats"
    return floats[0]


def _all_cohorts(root: Path) -> None:
    make_study(root, "camelyon")
    make_study(root, "tcga_4x4")
    make_study(root, "pcabiop", colour=False)


class TestTables:
    def test_one_table_per_tile_cohort_in_the_run_s_order_with_the_control_below(self, tmp_path):
        _all_cohorts(tmp_path)

        table = _float(_generator().build(tmp_path), "tab:sample-shortcut")

        rows = [
            line
            for line in table.splitlines()
            if line.endswith(r"\\") and not line.startswith(("&", "Model", r"\multicolumn"))
        ]
        names = [row.split(" & ")[0] for row in rows]
        assert names == ["Alpha", "Gamma", "Beta", "Delta", r"DINOv2-B$^{\dagger}$"]
        assert table.index("Delta") < table.index(r"\hline", table.index("Delta"))

    def test_an_encoder_below_the_flip_threshold_shows_dashes_and_the_caption_says_why(
        self, tmp_path
    ):
        _all_cohorts(tmp_path)

        table = _float(_generator().build(tmp_path), "tab:sample-shortcut")

        delta = next(line for line in table.splitlines() if line.startswith("Delta &"))
        cells = delta.removesuffix(r"\\").split(" & ")
        assert cells[1].strip() == "6"
        assert [c.strip() for c in cells[2:7]] == ["---"] * 5
        assert "fewer than 10 flipped" in table

    def test_cells_carry_the_point_and_its_interval(self, tmp_path):
        _all_cohorts(tmp_path)

        table = _float(_generator().build(tmp_path), "tab:sample-shortcut")

        alpha = next(line for line in table.splitlines() if line.startswith("Alpha &"))
        assert r"0.80\,{\scriptsize$[0.70, 0.90]$}" in alpha
        assert r"$0.20$\,{\scriptsize$[0.10, 0.30]$}" in alpha  # CRoMa - biology-only

    def test_the_footer_states_the_colour_classifier_and_conditional_auroc(self, tmp_path):
        _all_cohorts(tmp_path)

        table = _float(_generator().build(tmp_path), "tab:sample-shortcut-tcga4x4")

        assert "0.93" in table  # colour-only centre classifier balanced accuracy
        assert "0.69" in table  # median CRoMa AUROC within colour quintiles

    def test_pcabiop_gets_a_descriptive_table_without_colour(self, tmp_path):
        _all_cohorts(tmp_path)

        table = _float(_generator().build(tmp_path), "tab:sample-shortcut-pcabiop")

        assert "colour" not in table.lower()
        assert "Alpha &" in table

    def test_every_cohort_reports_the_margin_drop_correlations_tile_cohorts_also_colour(
        self, tmp_path
    ):
        _all_cohorts(tmp_path)
        tex = _generator().build(tmp_path)

        def last_cells(label):
            row = next(
                line for line in _float(tex, label).splitlines() if line.startswith("Alpha &")
            )
            return row.removesuffix(r" \\").split(" & ")[7:]

        assert last_cells("tab:sample-shortcut") == ["$-0.20$", "$-0.10$", "$-0.05$"]
        assert last_cells("tab:sample-shortcut-tcga4x4") == ["$-0.20$", "$-0.10$", "$-0.05$"]
        assert last_cells("tab:sample-shortcut-pcabiop") == ["$-0.20$", "$-0.10$"]

    def test_a_wide_table_is_shrunk_to_the_text_width_but_never_enlarged(self, tmp_path):
        _all_cohorts(tmp_path)

        tex = _generator().build(tmp_path)

        guard = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
        assert tex.count(guard) == 3

    def test_the_montage_caption_names_each_cohort_s_pick_and_its_flip_count(self, tmp_path):
        _all_cohorts(tmp_path)

        figure = _float(_generator().build(tmp_path), "fig:margin-montage")

        assert "Camelyon: Beta (AUROC 0.70), tumor, UMCU, $3$ of $11$ tiles flipped" in figure
        assert "TCGA-4x4: Beta" in figure

    def test_floats_follow_the_paper_placement_convention(self, tmp_path):
        _all_cohorts(tmp_path)

        tex = _generator().build(tmp_path)

        assert tex.count(r"\begin{table}[!htbp]") == 3
        assert tex.count(r"\begin{figure}[!htbp]") == 1

    def test_every_caption_is_within_the_350_word_limit(self, tmp_path):
        import re

        _all_cohorts(tmp_path)

        tex = _generator().build(tmp_path)

        captions = [line for line in tex.splitlines() if r"\caption{" in line]
        assert len(captions) == 4
        for caption in captions:
            assert len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", caption)) <= 350

    def test_rebuilding_is_byte_for_byte_identical(self, tmp_path):
        _all_cohorts(tmp_path)

        assert _generator().build(tmp_path) == _generator().build(tmp_path)

    def test_a_missing_cohort_fails_the_build(self, tmp_path):
        make_study(tmp_path, "camelyon")

        with pytest.raises(FileNotFoundError):
            _generator().build(tmp_path)


class TestMacros:
    def _macros(self, root):
        import re

        lines = _generator().macros(root)
        return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", "\n".join(lines), re.M))

    def test_each_cohort_gets_its_numbers_under_its_own_prefix(self, tmp_path):
        _all_cohorts(tmp_path)

        macros = self._macros(tmp_path)

        assert macros["CamelyonSampleShortcutCromaAurocMedian"] == "0.70"
        assert macros["CamelyonSampleShortcutNEvaluable"] == "3"
        assert macros["CamelyonSampleShortcutNCiAboveHalf"] == "2"
        assert macros["TcgaFourByFourSampleShortcutColourBacc"] == "0.93"
        assert macros["TcgaFourByFourSampleShortcutColourRhoMedian"] == "$-0.05$"
        assert macros["PcabiopSampleShortcutNeverHeldOut"] == "100"
        assert "PcabiopSampleShortcutColourBacc" not in macros

    def test_macro_names_are_letters_only_and_unique(self, tmp_path):
        _all_cohorts(tmp_path)

        names = [
            line.split("}")[0] for line in _generator().macros(tmp_path) if "newcommand" in line
        ]

        assert len(names) == len(set(names))
        assert all(name.removeprefix("\\newcommand{\\").isalpha() for name in names)

    def test_a_stale_study_fails_rather_than_dropping_its_macros(self, tmp_path):
        _all_cohorts(tmp_path)
        np.save(tmp_path / "output" / "embeddings" / "toy" / "Alpha.npy", np.ones(6))

        with pytest.raises(CaptionClaimError):
            _generator().macros(tmp_path)


def _figure_module():
    figures = str(ROOT / "scripts" / "repro" / "figures")
    if figures not in sys.path:
        sys.path.insert(0, figures)
    import margin_montage_figure

    return margin_montage_figure


class TestMarginMontageFigure:
    def test_each_tile_is_labelled_with_its_croma_flip_and_colour_confidence(self):
        row = pd.Series({"croma_m5": -0.3512, "flipped": True, "colour_confidence": 0.618})

        label = _figure_module().tile_label(row)

        assert label == "CRoMa −0.35\nflipped · colour 0.62"

    def test_an_unflipped_tile_says_so(self):
        row = pd.Series({"croma_m5": 0.2, "flipped": False, "colour_confidence": 0.5})

        assert _figure_module().tile_label(row) == "CRoMa 0.20\nnot flipped · colour 0.50"

    def test_renders_a_pdf_and_png_identically_twice(self, tmp_path):
        _all_cohorts(tmp_path)
        figure = _figure_module()

        first = figure.render(tmp_path, tmp_path / "plots", n=3)
        payload = first.read_bytes()
        again = figure.render(tmp_path, tmp_path / "plots", n=3)

        assert first == tmp_path / "plots" / "pdf" / "margin_montage.pdf"
        assert (tmp_path / "plots" / "png" / "margin_montage.png").is_file()
        assert again.read_bytes() == payload
