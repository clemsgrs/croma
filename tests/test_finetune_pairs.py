"""The robustness fine-tune vs parent study and the table and macros it feeds.

Seams under test: the study's pair derivation and paired measures
(``scripts/studies/finetune_pairs.py``), and the float basis that renders
``tab:finetune-pairs`` and the ``\\FineTune...`` macros (``scripts/repro/_finetune_pairs.py``
and ``generate_finetune_pairs_table.py``). Rendering is tested against temporary paths;
nothing here needs ``paper/`` or ``output/``.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts" / "studies", ROOT / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import finetune_pairs as study  # noqa: E402


def _metadata(**overrides) -> pd.DataFrame:
    """A model_metadata.csv-shaped frame: two fine-tunes, a distilled student, a control."""
    rows = [
        # model, panel, panel_order, parent_model, variant_role, benchmark_selection
        ("Base-A", "tile", 1, "", "", ""),
        ("Base-B", "tile", 2, "", "", ""),
        ("Teacher", "tile", 3, "", "teacher", ""),
        ("Student", "tile", 4, "Teacher", "distilled-student", ""),
        ("Robust-B", "tile", 5, "Base-B", "robustness-finetune", "pathorob-ri"),
        ("Robust-A", "tile", 6, "Base-A", "robustness-finetune", ""),
        ("DINOv2-B", "tile", 7, "", "", ""),
        ("Slide-Model", "slide", 1, "", "", ""),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "model",
            "panel",
            "panel_order",
            "parent_model",
            "variant_role",
            "benchmark_selection",
        ],
    )
    frame["adaptation_method_source"] = ""
    frame["adaptation_data_source"] = ""
    frame["corpus_domains"] = ""
    frame["institutional_domains"] = ""
    for column, values in overrides.items():
        frame[column] = values
    return frame


class TestPairsFromMetadata:
    def test_every_robustness_finetune_is_paired_with_its_declared_parent(self):
        pairs = study.finetune_pairs(_metadata())

        assert [(p.finetune, p.parent) for p in pairs] == [
            ("Robust-B", "Base-B"),
            ("Robust-A", "Base-A"),
        ]

    def test_a_finetune_without_a_declared_parent_fails(self):
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Robust-A", "parent_model"] = ""

        with pytest.raises(ValueError, match="Robust-A"):
            study.finetune_pairs(metadata)

    def test_a_parent_absent_from_the_tile_panel_fails(self):
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Robust-A", "parent_model"] = "Slide-Model"

        with pytest.raises(ValueError, match="Slide-Model"):
            study.finetune_pairs(metadata)

    def test_disclosure_carries_its_source_and_an_empty_cell_is_undisclosed(self):
        metadata = _metadata()
        robust_b = metadata["model"] == "Robust-B"
        metadata.loc[robust_b, "adaptation_method_source"] = "smith2026robust"
        metadata.loc[robust_b, "adaptation_data_source"] = " "

        by_name = {p.finetune: p for p in study.finetune_pairs(metadata)}

        assert by_name["Robust-B"].method_source == "smith2026robust"
        assert by_name["Robust-B"].data_source is None
        assert by_name["Robust-A"].method_source is None
        assert by_name["Robust-A"].data_source is None

    def test_benchmark_selection_is_carried_as_disclosed(self):
        by_name = {p.finetune: p for p in study.finetune_pairs(_metadata())}

        assert by_name["Robust-B"].benchmark_selection == "pathorob-ri"
        assert by_name["Robust-A"].benchmark_selection is None

    def test_the_committed_metadata_reads_without_mangling_literal_na(self):
        """``n/a`` is a disclosure marker, not a missing value; pandas eats it by default."""
        metadata = study.read_metadata()

        assert not metadata.isna().any().any()
        assert {"adaptation_method_source", "adaptation_data_source"} <= set(metadata.columns)
        assert all(p.parent in set(metadata["model"]) for p in study.finetune_pairs(metadata))


# --- paired representation measures on one benchmark run --------------------------------

PARENT = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SHIFT = 0.1


def _per_sample(values: dict[str, list[float]]) -> pd.DataFrame:
    """Per-sample CRoMa(m=5) rows: one slide per two samples, nested in class x centre."""
    frames = []
    for model, croma in values.items():
        n = len(croma)
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "sample_id": [f"s{i}" for i in range(n)],
                    "group_id": [f"g{i // 2}" for i in range(n)],
                    "label": ["a" if i < n // 2 else "b" for i in range(n)],
                    "confounder": ["X" if i < n // 2 else "Y" for i in range(n)],
                    "croma_m5": croma,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _metrics(**overrides) -> pd.DataFrame:
    """A metrics.csv-shaped frame whose CRoMa summaries match ``_per_sample``."""
    frame = pd.DataFrame(
        {
            "model": ["Base-A", "Robust-A", "Other", "DINOv2-B"],
            "croma": [0.15, 0.25, 0.20, 0.90],
            "croma_f0": [0.375, 0.25, 0.30, 0.0],
            "croma_ltm_alpha": [-0.2, -0.1, -0.15, 0.5],
            "ri": [0.40, 0.70, 0.5, 0.9],
            "mari": [0.35, 0.75, 0.5, 0.9],
            "bio_knn_bacc": [0.950, 0.940, 0.9, 0.8],
            "k": [11, 11, 11, 11],
            "evaluation_design": ["all"] * 4,
            "croma_m": [5] * 4,
            "croma_alpha": [0.1] * 4,
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


def _pair_a() -> list:
    return [study.FinetunePair("Robust-A", "Base-A")]


def _represent(per_sample=None, metrics=None, pairs=None, **kwargs) -> pd.DataFrame:
    per_sample = (
        _per_sample({"Base-A": PARENT, "Robust-A": [v + SHIFT for v in PARENT]})
        if per_sample is None
        else per_sample
    )
    return study.paired_representation(
        _pair_a() if pairs is None else pairs,
        _metrics() if metrics is None else metrics,
        per_sample,
        benchmark="toy",
        strata=("label", "confounder"),
        n_boot=kwargs.pop("n_boot", 60),
        seed=kwargs.pop("seed", 7),
        **kwargs,
    )


class TestPairedRepresentation:
    def test_a_parent_missing_from_the_run_roster_fails_loudly(self):
        metrics = _metrics().query("model != 'Base-A'")

        with pytest.raises(ValueError, match=r"Base-A.*toy"):
            _represent(metrics=metrics)

    def test_a_finetune_missing_from_the_run_roster_fails_loudly(self):
        metrics = _metrics().query("model != 'Robust-A'")

        with pytest.raises(ValueError, match=r"Robust-A.*toy"):
            _represent(metrics=metrics)

    def test_parent_finetune_and_difference_for_each_score(self):
        (row,) = _represent().to_dict("records")

        assert row["parent_croma"] == pytest.approx(0.15)
        assert row["finetune_croma"] == pytest.approx(0.25)
        assert row["delta_croma"] == pytest.approx(0.10)
        assert row["delta_f0"] == pytest.approx(-0.125)
        assert row["delta_ltm"] == pytest.approx(0.10)
        assert row["delta_ri"] == pytest.approx(0.30)
        assert row["delta_mari"] == pytest.approx(0.40)
        assert row["delta_bio_bacc"] == pytest.approx(-0.010)
        assert row["k"] == 11

    def test_one_draw_is_shared_so_a_uniform_shift_has_a_degenerate_interval(self):
        """Fine-tune = parent + 0.1 on every sample, so any *shared* draw moves the median
        and the lower-tail mean by exactly 0.1. Independent draws would not."""
        (row,) = _represent(n_boot=200).to_dict("records")

        for stat in ("croma", "ltm"):
            assert row[f"delta_{stat}_lo"] == pytest.approx(SHIFT)
            assert row[f"delta_{stat}_hi"] == pytest.approx(SHIFT)
        assert row["delta_f0_lo"] <= row["delta_f0"] <= row["delta_f0_hi"]
        assert row["delta_f0_lo"] < row["delta_f0_hi"]

    def test_ranks_are_by_median_croma_over_the_ranked_panel(self):
        """The natural-image control leads this toy run but holds no rank."""
        (row,) = _represent().to_dict("records")

        assert (row["parent_rank"], row["finetune_rank"]) == (3, 1)
        assert row["rank_change"] == 2
        assert row["n_ranked"] == 3

    def test_summaries_that_disagree_with_the_run_fail(self):
        metrics = _metrics(croma=[0.5, 0.25, 0.20, 0.90])

        with pytest.raises(ValueError, match="Base-A"):
            _represent(metrics=metrics)

    def test_only_the_all_design_is_accepted(self):
        metrics = _metrics(evaluation_design=["paired_2x2"] * 4)

        with pytest.raises(ValueError, match="all"):
            _represent(metrics=metrics)

    def test_the_draw_is_reproducible_from_its_seed(self):
        first = _represent(seed=3, n_boot=40)
        again = _represent(seed=3, n_boot=40)

        pd.testing.assert_frame_equal(first, again)


def _consistent_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Setup only: a metrics frame whose pooled CRoMa summaries match ``per_sample``."""
    from croma.metrics.tail import compute_tail_metrics

    rows = []
    for model, group in per_sample.groupby("model"):
        values = group["croma_m5"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "croma": float(np.median(values)),
                "croma_f0": float((values <= 0).mean()),
                "croma_ltm_alpha": compute_tail_metrics(values, alpha=0.1).ltm_alpha,
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("ri", "mari", "bio_knn_bacc"):
        frame[column] = 0.5
    frame["k"], frame["evaluation_design"] = 11, "all"
    frame["croma_m"], frame["croma_alpha"] = 5, 0.1
    return frame


class TestPerSampleShifts:
    PARENT = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    FINETUNE = [-0.3, -0.5, 0.1, -0.1, 0.2, 0.0, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8]

    def _row(self) -> dict:
        per_sample = _per_sample({"Base-A": self.PARENT, "Robust-A": self.FINETUNE})
        # Row order must not matter: the two encoders are joined by sample_id.
        per_sample = per_sample.sample(frac=1.0, random_state=0).reset_index(drop=True)
        (row,) = _represent(per_sample=per_sample, metrics=_consistent_metrics(per_sample)).to_dict(
            "records"
        )
        return row

    def test_share_of_samples_whose_croma_rises(self):
        assert self._row()["share_rises"] == pytest.approx(9 / 12)

    def test_median_shift_among_the_parents_lowest_decile(self):
        """The parent's lowest decile is s0 and s1 (shifts +0.2 and -0.1), not the
        fine-tune's, whose lowest samples are s1 and s0 in the other order."""
        row = self._row()

        assert row["parent_low_decile_n"] == 2
        assert row["parent_low_decile_median_shift"] == pytest.approx(0.05)

    def test_share_of_the_parents_confounder_dominant_samples_that_turn_positive(self):
        row = self._row()

        assert row["parent_nonpositive_n"] == 5
        assert row["parent_nonpositive_turn_positive"] == pytest.approx(2 / 5)


# --- paired downstream shortcut susceptibility (nIPD) -----------------------------------

#: Two Cramer's-V points, chance 0.5. A fall from 0.9 to 0.7 halves the skill above
#: chance, so g = [0, -0.5] and nIPD is the trapezoid area -0.25.
V = [0.0, 1.0]
CHANCE = 0.5
#: Three probe runs per Cramer's-V point, with run-to-run spread.
FALLS_HALF = [[0.88, 0.90, 0.92], [0.66, 0.70, 0.74]]
FALLS_QUARTER = [[0.88, 0.90, 0.92], [0.76, 0.80, 0.84]]


def _cell(accuracies: list[list[float]]) -> dict:
    """An APD study cell: the same matrix in and out of domain unless overridden."""
    return {"id_test_accuracies": accuracies, "ood_test_accuracies": accuracies}


def _apd_rows(**nipd) -> pd.DataFrame:
    """apd.csv rows; ``nipd`` maps model -> (nipd_id, nipd_ood)."""
    return pd.DataFrame([{"model": m, "nipd_id": i, "nipd_ood": o} for m, (i, o) in nipd.items()])


def _downstream(cells, apd_rows=None, **kwargs) -> dict:
    apd_rows = (
        _apd_rows(**{"Base-A": (-0.25, -0.25), "Robust-A": (-0.125, -0.125)})
        if apd_rows is None
        else apd_rows
    )
    (row,) = study.paired_downstream(
        _pair_a(),
        cells,
        apd_rows,
        benchmark="toy",
        cramers_v=V,
        chance=CHANCE,
        n_boot=kwargs.pop("n_boot", 200),
        seed=kwargs.pop("seed", 5),
        **kwargs,
    ).to_dict("records")
    return row


class TestPairedDownstream:
    def test_parent_finetune_and_difference_in_and_out_of_domain(self):
        row = _downstream({"Base-A": _cell(FALLS_HALF), "Robust-A": _cell(FALLS_QUARTER)})

        for domain in ("id", "ood"):
            assert row[f"parent_nipd_{domain}"] == pytest.approx(-0.25)
            assert row[f"finetune_nipd_{domain}"] == pytest.approx(-0.125)
            assert row[f"delta_nipd_{domain}"] == pytest.approx(0.125)
            assert row[f"delta_nipd_{domain}_lo"] <= 0.125 <= row[f"delta_nipd_{domain}_hi"]
        assert row["nipd_runs"] == 3

    def test_probe_runs_are_resampled_jointly_across_the_pair(self):
        """Run j of both encoders saw the same arrangement, so the two are resampled with
        the same run indices: identical matrices can only differ by exactly zero."""
        row = _downstream(
            {"Base-A": _cell(FALLS_HALF), "Robust-A": _cell(FALLS_HALF)},
            apd_rows=_apd_rows(**{"Base-A": (-0.25, -0.25), "Robust-A": (-0.25, -0.25)}),
        )

        assert (row["delta_nipd_id_lo"], row["delta_nipd_id_hi"]) == (0.0, 0.0)

    def test_a_missing_cell_fails_naming_the_encoder(self):
        with pytest.raises(ValueError, match=r"Robust-A.*toy"):
            _downstream({"Base-A": _cell(FALLS_HALF)})

    def test_a_cell_that_disagrees_with_the_study_table_fails(self):
        with pytest.raises(ValueError, match="Robust-A"):
            _downstream(
                {"Base-A": _cell(FALLS_HALF), "Robust-A": _cell(FALLS_QUARTER)},
                apd_rows=_apd_rows(**{"Base-A": (-0.25, -0.25), "Robust-A": (-0.3, -0.125)}),
            )

    def test_the_resample_is_reproducible_from_its_seed(self):
        cells = {"Base-A": _cell(FALLS_HALF), "Robust-A": _cell(FALLS_QUARTER)}

        assert _downstream(cells, seed=11) == _downstream(cells, seed=11)


# --- the float basis: loading the study and the \FineTune... macros ---------------------

import _finetune_pairs as basis  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

#: (fine-tune, parent) in the order ``_metadata`` lists the fine-tunes.
STUDY_PAIRS = [("Robust-B", "Base-B"), ("Robust-A", "Base-A")]


def _study_row(finetune: str, parent: str, benchmark: str, **overrides) -> dict:
    """One study CSV row with hand-picked values; every difference is stated, not derived."""
    row = {
        "benchmark": benchmark,
        "finetune": finetune,
        "parent": parent,
        "k": 11,
        "strata": "confounder" if benchmark == "pathorob-tolkach-esca" else "label+confounder",
        "n_boot": 2000,
        "level": 0.95,
        "seed": 12345,
        "n_ranked": 26,
        "parent_croma": 0.10,
        "finetune_croma": 0.28,
        "delta_croma": 0.18,
        "delta_croma_lo": 0.15,
        "delta_croma_hi": 0.204,
        "parent_f0": 0.400,
        "finetune_f0": 0.088,
        "delta_f0": -0.312,
        "delta_f0_lo": -0.355,
        "delta_f0_hi": -0.272,
        "parent_ltm": -0.40,
        "finetune_ltm": -0.07,
        "delta_ltm": 0.33,
        "delta_ltm_lo": 0.28,
        "delta_ltm_hi": 0.38,
        "parent_bio_bacc": 0.951,
        "finetune_bio_bacc": 0.956,
        "delta_bio_bacc": 0.005,
        "parent_rank": 13,
        "finetune_rank": 2,
        "rank_change": 11,
        "parent_nipd_id": -0.075,
        "finetune_nipd_id": -0.007,
        "delta_nipd_id": 0.068,
        "delta_nipd_id_lo": 0.039,
        "delta_nipd_id_hi": 0.106,
        "nipd_runs": 20,
        "parent_nipd_ood": 0.002,
        "finetune_nipd_ood": -0.003,
        "delta_nipd_ood": -0.005,
        "delta_nipd_ood_lo": -0.011,
        "delta_nipd_ood_hi": 0.001,
    }
    row.update(overrides)
    return row


def _study(**cell_overrides) -> pd.DataFrame:
    """The study CSV for ``STUDY_PAIRS``; ``cell_overrides[(finetune, benchmark)]`` edits one."""
    rows = []
    for finetune, parent in STUDY_PAIRS:
        for benchmark in study.BENCHMARKS:
            extra = cell_overrides.get(f"{finetune}|{benchmark}", {})
            rows.append(_study_row(finetune, parent, benchmark, **extra))
    return pd.DataFrame(rows)


class TestLoad:
    def test_a_study_that_was_never_run_fails_with_its_remedy(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="finetune_pairs.py"):
            basis.load(tmp_path / "absent.csv", metadata=_metadata())

    def test_a_study_that_no_longer_matches_the_metadata_pairs_fails(self, tmp_path):
        path = tmp_path / "finetune_pairs.csv"
        _study().to_csv(path, index=False)
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Student", "variant_role"] = "robustness-finetune"

        with pytest.raises(CaptionClaimError, match="Student"):
            basis.load(path, metadata=metadata)

    def test_a_current_study_loads_in_pair_order(self, tmp_path):
        path = tmp_path / "finetune_pairs.csv"
        _study().to_csv(path, index=False)

        frame = basis.load(path, metadata=_metadata())

        assert frame["finetune"].tolist() == ["Robust-B"] * 3 + ["Robust-A"] * 3


class TestMacros:
    def _macros(self, frame=None) -> dict[str, str]:
        lines = basis.macro_lines(_study() if frame is None else frame)
        parsed = [re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", line) for line in lines]
        assert all(parsed), lines
        return {m.group(1): m.group(2) for m in parsed}

    def test_each_difference_is_a_signed_macro_per_pair_and_cohort(self):
        macros = self._macros()

        assert macros["FineTuneRobustACamelyonCromaDelta"] == "$+0.18$"
        assert macros["FineTuneRobustBTcgaFourByFourFZeroDelta"] == "$-0.312$"
        assert macros["FineTuneRobustATolkachLtmDelta"] == "$+0.33$"
        assert macros["FineTuneRobustACamelyonBioBaccDelta"] == "$+0.005$"
        assert macros["FineTuneRobustACamelyonNipdIdDelta"] == "$+0.068$"
        assert macros["FineTuneRobustACamelyonCromaDeltaCi"] == "$[0.15, 0.20]$"

    def test_counts_over_every_pair_and_cohort(self):
        frame = _study(
            **{"Robust-A|pathorob-tolkach-esca": {"delta_f0": 0.010, "delta_ltm": -0.02}}
        )

        macros = self._macros(frame)

        assert macros["FineTuneNCells"] == "6"
        assert macros["FineTuneFZeroFallsCount"] == "5"
        assert macros["FineTuneLtmRisesCount"] == "5"

    def test_macro_names_hold_letters_only(self):
        frame = _study().replace({"Robust-A": "Robust-2"})

        assert "FineTuneRobustTwoCamelyonCromaDelta" in self._macros(frame)


# --- the rendered table ------------------------------------------------------------------


def _render(frame=None, metadata=None, bib_keys=frozenset()) -> str:
    import generate_finetune_pairs_table as table

    return table.build(
        _study() if frame is None else frame,
        _metadata() if metadata is None else metadata,
        bib_keys=set(bib_keys),
    )


def _caption(tex: str) -> str:
    (line,) = [line for line in tex.splitlines() if line.startswith(r"\caption{")]
    return line


def _block(tex: str, finetune: str) -> str:
    """The rows of one pair's block, from its header to the next header or rule."""
    lines = tex.splitlines()
    start = next(i for i, line in enumerate(lines) if "multicolumn" in line and finetune in line)
    end = next(
        i
        for i in range(start + 1, len(lines))
        if "multicolumn" in lines[i] or lines[i] == r"\hline"
    )
    return "\n".join(lines[start:end])


class TestTable:
    def test_one_block_per_pair_with_one_row_per_cohort(self):
        tex = _render()

        assert r"\label{tab:finetune-pairs}" in tex
        assert tex.startswith(r"\begin{table}[!htbp]")
        for finetune, parent in STUDY_PAIRS:
            block = _block(tex, finetune)
            assert rf"{parent} $\rightarrow$ {finetune}" in block
            for cohort in ("Camelyon", "TCGA-4x4", "Tolkach-ESCA"):
                assert sum(line.startswith(cohort) for line in block.splitlines()) == 1

    def test_each_cohort_shows_parent_to_finetune_then_the_difference_then_its_interval(self):
        rows = _block(_render(), "Robust-A").splitlines()
        i = next(i for i, line in enumerate(rows) if line.startswith("Camelyon"))
        values, delta, ci = (rows[i + j].split(" & ") for j in range(3))

        assert values[1:3] == [r"$0.10 \to 0.28$", r"$0.400 \to 0.088$"]
        assert values[-1] == r"$13 \to 2$ \\"
        assert delta[1:6] == ["$+0.18$", "$-0.312$", "$+0.33$", "$+0.005$", "$+0.068$"]
        assert delta[-1] == r"$+11$ \\"
        assert ci[1:3] == ["$[0.15, 0.20]$", "$[-0.355, -0.272]$"]
        assert ci[4:7] == ["", "$[0.039, 0.106]$", "$[-0.011, 0.001]$"]

    def test_the_caption_is_one_line_within_the_journal_word_limit(self):
        caption = _caption(_render())

        assert caption.endswith("}")
        assert len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", caption)) <= 350

    def test_the_caption_states_the_pairs_come_from_metadata_and_how_intervals_are_drawn(self):
        caption = _caption(_render())

        assert "model metadata" in caption
        assert "B{=}2000" in caption
        assert "$20$ probe runs" in caption
        assert "26" in caption


class TestDisclosureCaption:
    def test_blank_disclosure_cells_read_undisclosed(self):
        caption = _caption(_render())

        assert "undisclosed for Robust-B and Robust-A" in caption

    def test_a_disclosed_source_is_cited_and_a_url_is_linked(self):
        metadata = _metadata()
        robust_b = metadata["model"] == "Robust-B"
        metadata.loc[robust_b, "adaptation_method_source"] = "smith2026robust"
        metadata.loc[robust_b, "adaptation_data_source"] = "https://example.org/data"

        caption = _caption(_render(metadata=metadata, bib_keys={"smith2026robust"}))

        assert r"\cite{smith2026robust}" in caption
        assert r"\url{https://example.org/data}" in caption
        assert "undisclosed for Robust-A" in caption

    def test_a_citation_key_absent_from_the_bibliography_fails(self):
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Robust-B", "adaptation_method_source"] = "nokey2026"

        with pytest.raises(CaptionClaimError, match="nokey2026"):
            _render(metadata=metadata)


class TestMarkers:
    def test_a_benchmark_selected_finetune_carries_the_double_dagger_and_its_legend(self):
        tex = _render()

        assert r"Robust-B$^{\ddagger}$" in _block(tex, "Robust-B")
        assert r"\ddagger" not in _block(tex, "Robust-A")
        assert "PathoROB" in _caption(tex) and "checkpoint selection" in _caption(tex)

    def test_an_undocumented_kind_of_benchmark_selection_fails(self):
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Robust-B", "benchmark_selection"] = "other-benchmark"

        with pytest.raises(CaptionClaimError, match="other-benchmark"):
            _render(metadata=metadata)

    def test_a_cohort_in_the_pairs_pretraining_provenance_is_daggered_and_named(self):
        metadata = _metadata()
        exposed = metadata["model"].isin(["Base-A", "Robust-A"])
        metadata.loc[exposed, "corpus_domains"] = "tcga"

        tex = _render(metadata=metadata)
        block_a, block_b = _block(tex, "Robust-A"), _block(tex, "Robust-B")

        assert r"TCGA-4x4$^{\dagger}$" in block_a
        assert r"Camelyon$^{\dagger}$" not in block_a
        assert r"\dagger" not in block_b.replace(r"\ddagger", "")
        assert "TCGA-4x4 for Base-A and Robust-A" in _caption(tex)

    def test_without_exposure_no_dagger_is_drawn(self):
        assert r"\dagger" not in _render().replace(r"\ddagger", "")
