r"""The split-schedule table documents the confounder-biased probe protocol (``tab:split-schedule``).

Each benchmark's schedule walks the training composition from balanced to fully confounded.
The table states, per split, Cramér's V, the training slides per favourable and unfavourable
cell, the training and validation sizes and the held-out slides per cell. The schedules and
the slicing come from the code the sweep itself runs, so the numbers below are worked out
by hand from PathoROB's published Camelyon schedule and croma's PCaBiop schedule.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts/repro", ROOT / "scripts/bench", ROOT / "scripts/studies/apd"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _split_schedule as schedule  # noqa: E402
import generate_split_schedule_table as table  # noqa: E402


def _metadata(cells: dict[tuple[str, str], int], tiles: int, ood: dict[tuple[str, str], int]):
    """A PathoROB-shaped APD metadata frame: ``slides`` slides of ``tiles`` rows per cell."""
    rows = []
    for subset, spec in (("ID", cells), ("OOD", ood)):
        for (centre, label), slides in spec.items():
            for s in range(slides):
                for t in range(tiles):
                    rows.append((subset, f"{subset}-{centre}-{label}-{s}", t, label, centre))
    return pd.DataFrame(
        rows, columns=["subset", "slide_id", "patch_id", "biological_class", "medical_center"]
    )


@pytest.fixture(scope="module")
def camelyon_metadata():
    cells = {(c, b): 17 for c in ("RUMC", "UMCU") for b in ("normal", "tumor")}
    ood = {(c, b): 1 for c in ("CWZ", "RST", "LPON") for b in ("normal", "tumor")}
    return _metadata(cells, 300, ood)


@pytest.fixture(scope="module")
def pcabiop_metadata():
    cells = {(c, b): 250 for c in ("radboud", "karolinska") for b in ("benign", "cancer")}
    ood = {("PAR", b): 162 for b in ("benign", "cancer")}
    return _metadata(cells, 1, ood)


def test_camelyon_walks_pathorobs_schedule_from_balanced_to_confounded(camelyon_metadata):
    camelyon = schedule.schedule("camelyon", camelyon_metadata)

    assert len(camelyon.splits) == 8
    first, last = camelyon.splits[0], camelyon.splits[-1]
    # Balanced: 7 slides in every cell; 2,100 tiles x 4 cells; one 150-tile validation
    # slice per cell (2,100 / 14); the held-out tail starts at slide 15 of 17.
    assert (first.favourable, first.unfavourable) == ((7,), 7)
    assert (first.train, first.validation, first.test_slides) == (8400, 600, (2, 2))
    assert first.cramers_v == pytest.approx(0.0)
    # Fully confounded: the favourable cells are empty, the training size is conserved.
    assert (last.favourable, last.unfavourable) == ((0,), 14)
    assert (last.train, last.validation, last.test_slides) == (8400, 600, (2, 2))
    assert last.cramers_v == pytest.approx(1.0)


def test_pools_count_the_in_domain_and_held_out_centres(camelyon_metadata):
    camelyon = schedule.schedule("camelyon", camelyon_metadata)

    assert camelyon.tiles_per_slide == 300
    assert camelyon.n_classes == 2
    assert camelyon.id_pool == schedule.Pool(("RUMC", "UMCU"), slides=68, tiles=20400)
    assert camelyon.ood_pool == schedule.Pool(("CWZ", "RST", "LPON"), slides=6, tiles=1800)
    assert (camelyon.replicates, camelyon.seed) == (20, 1000)
    assert not camelyon.croma_authored


def test_pcabiop_uses_the_explicit_slide_level_validation_fraction(pcabiop_metadata):
    pcabiop = schedule.schedule("pcabiop", pcabiop_metadata)

    assert len(pcabiop.splits) == 11
    first = pcabiop.splits[0]
    # 100 slides per cell; a 10% validation slice (10 per cell); the tail starts after
    # 200 + 20 + 1 slides, leaving 29 of 250 per cell.
    assert (first.favourable, first.unfavourable) == ((100,), 100)
    assert (first.train, first.validation, first.test_slides) == (400, 40, (29, 29))
    assert pcabiop.croma_authored
    assert pcabiop.ood_pool == schedule.Pool(("PAR",), slides=324, tiles=324)


def test_a_tiles_per_slide_override_thins_training_but_keeps_the_full_id_pool(camelyon_metadata):
    reduced = schedule.schedule("camelyon", camelyon_metadata, tiles_per_slide=30)

    assert reduced.tiles_per_slide == 30
    # The reduced sweep scores the held-out tail at full tiles, so the pool it draws on is
    # the cohort's: 68 slides of 300 tiles, not of 30.
    assert reduced.id_pool.tiles == 20400
    assert [s.train for s in reduced.splits] == [840] * 8
    assert all(s.test_slides == (2, 2) for s in reduced.splits)
    assert [s.unfavourable for s in reduced.splits] == list(range(7, 15))


def test_a_reduced_tile_override_validates_as_the_thinned_sweep_does(camelyon_metadata):
    # The reduced sweep keeps the full sweep's slices and shrinks each cell's validation
    # slice by the training factor, rounding down but never below one row. Split 3 trains
    # 4 slides per favourable cell and 10 per unfavourable one; at full tiles they validate
    # on 1,200 / 14 = 85 and 3,000 / 14 = 214 rows, so at 3 tiles on 1 and 2 rows each.
    reduced = schedule.schedule("camelyon", camelyon_metadata, tiles_per_slide=3)

    split = reduced.splits[3]
    assert (split.favourable, split.unfavourable) == ((4,), 10)
    assert (split.train, split.validation) == ((2 * 4 + 2 * 10) * 3, 2 * 1 + 2 * 2)
    # The held-out tail is the full sweep's.
    assert split.test_slides == (2, 2)


def _tcga_metadata(tiles: int):
    cells = {(c, b): 12 for c in range(4) for b in range(4)}
    centres = ["Asterand", "Christiana Healthcare", "Roswell Park", "University of Pittsburgh"]
    classes = [
        "Breast_invasive_carcinoma",
        "Colon_adenocarcinoma",
        "Lung_adenocarcinoma",
        "Lung_squamous_cell_carcinoma",
    ]
    held_out = ["Cureline", "Greater Poland Cancer Center", "International Genomics Consortium"]
    return _metadata(
        {(centres[c], classes[b]): n for (c, b), n in cells.items()},
        tiles,
        {(centre, classes[0]): 1 for centre in [*held_out, "Johns Hopkins"]},
    )


def test_a_one_tile_override_keeps_a_validation_row_in_every_cell():
    # At one tile per slide a cell's full-tile validation slice shrinks below one row, and
    # the sweep keeps one: the schedule never runs out of validation rows.
    tcga = schedule.schedule("tcga_4x4", _tcga_metadata(30), tiles_per_slide=1)

    assert all(s.train == 32 for s in tcga.splits)
    assert tcga.splits[0].validation == 16


def test_tcga_lists_both_favourable_cell_types():
    md = _tcga_metadata(30)

    tcga = schedule.schedule("tcga_4x4", md)

    # Diagonal cells grow (split + 2); the anti-diagonal and the other off-diagonal cells
    # shrink on different trajectories, so the favourable column carries both.
    assert [s.unfavourable for s in tcga.splits] == [2, 3, 4, 5, 6, 7, 8]
    assert [s.favourable for s in tcga.splits][:3] == [(2, 2), (2, 1), (1, 2)]
    assert tcga.favourable_types == ("off-diagonal", "anti-diagonal")
    assert all(s.train == 32 * 30 for s in tcga.splits)


def test_the_stated_c_grid_is_the_vendored_probes():
    from croma.downstream import _pathorob

    assert "np.linspace(-8, 4, 15)" in inspect.getsource(_pathorob.train_logistic_regression)
    assert table.C_GRID == (-8, 4, 15)


def test_the_float_states_the_probe_once_and_footnotes_pcabiop(
    monkeypatch, camelyon_metadata, pcabiop_metadata
):
    frames = {"camelyon": camelyon_metadata, "pcabiop": pcabiop_metadata}
    monkeypatch.setattr(
        table,
        "schedules",
        lambda root, tiles_per_slide=None: [
            schedule.schedule(key, frames[key], tiles_per_slide=(tiles_per_slide or {}).get(key))
            for key in frames
        ],
    )

    tex = table.build(ROOT)

    assert tex.startswith(r"\begin{table}[!htbp]")
    assert r"\label{tab:split-schedule}" in tex
    caption = next(line for line in tex.splitlines() if line.startswith(r"\caption"))
    assert "logistic regression" in caption and "balanced accuracy" in caption
    assert r"$[10^{-8}, 10^{4}]$" in caption and "15 log-spaced" in caption
    assert caption.count("logistic regression") == 1
    # One row per split under each benchmark, and the croma-authored schedule is marked.
    assert tex.count(r"\\") >= 8 + 11
    assert r"PCaBiop$^{a}$" in tex
    assert "validation fraction" in tex.split(r"\end{tabular}")[-1]


def test_the_float_takes_a_tiles_per_slide_override(monkeypatch, camelyon_metadata):
    monkeypatch.setattr(
        table,
        "schedules",
        lambda root, tiles_per_slide=None: [
            schedule.schedule(
                "camelyon",
                camelyon_metadata,
                tiles_per_slide=(tiles_per_slide or {}).get("camelyon"),
            )
        ],
    )

    full = table.build(ROOT)
    reduced = table.build(ROOT, tiles_per_slide={"camelyon": 30}, label="tab:split-schedule-30")

    assert "8{,}400" in full and "8{,}400" not in reduced
    assert r"\label{tab:split-schedule-30}" in reduced


def _camelyon_only(monkeypatch, camelyon_metadata):
    monkeypatch.setattr(
        table,
        "schedules",
        lambda root, tiles_per_slide=None: [
            schedule.schedule(
                "camelyon",
                camelyon_metadata,
                tiles_per_slide=(tiles_per_slide or {}).get("camelyon"),
            )
        ],
    )


def _panel_c(tex: str) -> list[list[str]]:
    panel = tex.split(r"\emph{(c)")[1].split(r"\end{tabular}")[0]
    rows = [line for line in panel.splitlines()[3:] if line.endswith(r"\\")]
    return [line.removesuffix(r" \\").split(" & ") for line in rows]


def test_the_float_lists_the_reduced_tile_sweeps_the_records_ran(monkeypatch, camelyon_metadata):
    _camelyon_only(monkeypatch, camelyon_metadata)

    tex = table.build(ROOT)

    # Camelyon was recorded at 30 and 3 of its 300 tiles per slide: 28 training slides per
    # split, and each cell's full-tile validation slice shrunk by 10 and by 100, never
    # below one row (worked per split in the override tests above).
    assert _panel_c(tex) == [["Camelyon", "30", "840", "58--60"], ["", "3", "84", "4--6"]]
    caption = next(line for line in tex.splitlines() if line.startswith(r"\caption"))
    assert r"\textbf{c,}" in caption and "never below one row" in caption


def _caption_of(monkeypatch, made) -> str:
    monkeypatch.setattr(table, "schedules", lambda root, tiles_per_slide=None: made)
    tex = table.build(ROOT, tiles_per_slide={"any": 1}, label="tab:split-schedule-reduced")
    return next(line for line in tex.splitlines() if line.startswith(r"\caption"))


def test_a_one_tile_override_does_not_make_a_tile_cohort_slide_level(monkeypatch, pcabiop_metadata):
    made = [
        schedule.schedule("tcga_4x4", _tcga_metadata(30), tiles_per_slide=1),
        schedule.schedule("pcabiop", pcabiop_metadata),
    ]

    caption = _caption_of(monkeypatch, made)

    assert " PCaBiop is slide-level, with one embedding per slide" in caption
    assert "TCGA-4x4 and" not in caption


def test_a_reduced_case_block_keeps_its_tiles_and_trains_on_some(monkeypatch, camelyon_metadata):
    # A case-based cohort under an override: the block is still the cohort's 300 tiles.
    made = [
        dataclasses.replace(
            schedule.schedule("camelyon", camelyon_metadata, tiles_per_slide=30), case_based=True
        )
    ]

    caption = _caption_of(monkeypatch, made)

    assert "one case's 300-tile block within a class, of which only 30 tiles train," in caption
    full = dataclasses.replace(schedule.schedule("camelyon", camelyon_metadata), case_based=True)
    assert "one case's 300-tile block within a class, since" in _caption_of(monkeypatch, [full])


def test_a_reduced_variant_has_no_reduced_tile_panel(monkeypatch, camelyon_metadata):
    _camelyon_only(monkeypatch, camelyon_metadata)

    tex = table.build(ROOT, tiles_per_slide={"camelyon": 30}, label="tab:split-schedule-30")

    assert r"\emph{(c)" not in tex and r"\textbf{c,}" not in tex


@pytest.mark.skipif(
    not (ROOT / "data/pathorob/metadata/camelyon.csv").exists(),
    reason="the APD metadata lives under data/, which is not tracked",
)
def test_the_live_schedules_cover_the_four_benchmarks():
    live = schedule.schedules(ROOT)

    assert [s.name for s in live] == ["Camelyon", "TCGA-4x4", "Tolkach-ESCA", "PCaBiop"]
    assert [len(s.splits) for s in live] == [8, 7, 4, 11]
    tolkach = live[2]
    assert tolkach.n_classes == 6
    assert tolkach.splits[0].test_slides == (2, 2)


@pytest.mark.skipif(
    not (ROOT / "data/pathorob/metadata/camelyon.csv").exists(),
    reason="the APD metadata lives under data/, which is not tracked",
)
def test_a_live_override_reports_each_cohorts_full_id_pool():
    # The probe records hold these ID rows at every recorded tile level.
    reduced = schedule.schedules(
        ROOT, tiles_per_slide={"camelyon": 30, "tcga_4x4": 1, "tolkach": 10}
    )

    assert [s.id_pool.tiles for s in reduced] == [20400, 5760, 10800, 1000]


def test_an_output_path_outside_the_repository_is_written(monkeypatch, tmp_path):
    out = tmp_path / "x.tex"
    monkeypatch.setattr(table, "build", lambda tiles_per_slide=None, label=table.LABEL: "tex\n")
    monkeypatch.setattr(sys, "argv", ["generate_split_schedule_table.py", "--out", str(out)])

    assert table.main() == 0
    assert out.read_text() == "tex\n"

    monkeypatch.setattr(
        sys, "argv", ["generate_split_schedule_table.py", "--out", str(out), "--check"]
    )
    assert table.main() == 0
    out.write_text("old\n")
    assert table.main() == 1
