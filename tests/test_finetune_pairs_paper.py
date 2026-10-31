"""The robustness fine-tune vs parent study's paper surfaces: its basis, the table, the macros.

Seams under test: the float basis that loads the study CSV and emits the ``\\FineTune...``
macros (``scripts/repro/_finetune_pairs.py``) and the ``tab:finetune-pairs`` generator
(``generate_finetune_pairs_table.py``). Everything renders from synthetic frames and inputs
under ``tmp_path``; nothing here needs ``paper/`` or ``output/``. The paper tooling is not
in the source distribution (ADR-0012), so neither is this file; the study itself is tested
in ``test_finetune_pairs_study.py``.
"""

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts" / "studies", ROOT / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import finetune_pairs as study  # noqa: E402
import _finetune_pairs as basis  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402


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


# --- the float basis: loading the study and the \FineTune... macros ---------------------

#: (fine-tune, parent) in the order ``_metadata`` lists the fine-tunes.
STUDY_PAIRS = [("Robust-B", "Base-B"), ("Robust-A", "Base-A")]
#: The study columns that record the sha256 of the inputs a row was computed from.
FINGERPRINT_COLUMNS = ["metrics_sha256", "apd_sha256"]


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
        "metrics_sha256": "0" * 64,
        "apd_sha256": "0" * 64,
    }
    row.update(overrides)
    return row


def _study(*, fingerprints=None, **cell_overrides) -> pd.DataFrame:
    """The study CSV for ``STUDY_PAIRS``; ``cell_overrides[(finetune, benchmark)]`` edits one.

    ``fingerprints[benchmark]`` holds the input fingerprints of that cohort's rows.
    """
    rows = []
    for finetune, parent in STUDY_PAIRS:
        for benchmark in study.BENCHMARKS:
            extra = dict((fingerprints or {}).get(benchmark, {}))
            extra.update(cell_overrides.get(f"{finetune}|{benchmark}", {}))
            rows.append(_study_row(finetune, parent, benchmark, **extra))
    return pd.DataFrame(rows)


def _metrics_csv(root: Path, benchmark: str) -> Path:
    return root / by_benchmark(benchmark).metrics_rel


def _apd_csv(root: Path) -> Path:
    return root / "output" / "studies" / "apd" / "apd.csv"


def _write_inputs(root: Path) -> dict[str, dict[str, str]]:
    """Each cohort's ``metrics.csv`` and the APD study table under ``root``, where
    ``paper_manifest`` resolves them; returns each cohort's fingerprints."""

    def write(path: Path, text: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    apd = write(_apd_csv(root), "dataset,model,nipd_id,nipd_ood\ncamelyon,Base-A,-0.1,0.0\n")
    return {
        benchmark: {
            "metrics_sha256": write(_metrics_csv(root, benchmark), f"model,k\n{benchmark},11\n"),
            "apd_sha256": apd,
        }
        for benchmark in study.BENCHMARKS
    }


def _current_study(root: Path) -> Path:
    """A study CSV under ``root`` whose fingerprints match the inputs written beside it."""
    path = root / "finetune_pairs.csv"
    _study(fingerprints=_write_inputs(root)).to_csv(path, index=False)
    return path


class TestLoad:
    def test_a_study_that_was_never_run_fails_with_its_remedy(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="finetune_pairs.py"):
            basis.load(tmp_path / "absent.csv", metadata=_metadata(), root=tmp_path)

    def test_a_study_that_no_longer_matches_the_metadata_pairs_fails(self, tmp_path):
        path = _current_study(tmp_path)
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Student", "variant_role"] = "robustness-finetune"

        with pytest.raises(CaptionClaimError, match="Student"):
            basis.load(path, metadata=metadata, root=tmp_path)

    def test_a_current_study_loads_in_pair_order(self, tmp_path):
        path = _current_study(tmp_path)

        frame = basis.load(path, metadata=_metadata(), root=tmp_path)

        assert frame["finetune"].tolist() == ["Robust-B"] * 3 + ["Robust-A"] * 3

    def test_a_benchmark_run_redone_after_the_study_fails(self, tmp_path):
        """Same pairs, new run contents (an encoder joined, the median k moved): the study's
        ranks, k and differences are stale even though its rows still match the metadata."""
        path = _current_study(tmp_path)
        metrics = _metrics_csv(tmp_path, "pathorob-tcga-4x4")
        metrics.write_text(metrics.read_text() + "Newcomer,11\n")

        with pytest.raises(CaptionClaimError, match=r"(?s)metrics\.csv.*finetune_pairs\.py"):
            basis.load(path, metadata=_metadata(), root=tmp_path)

    def test_an_apd_study_redone_after_the_study_fails(self, tmp_path):
        path = _current_study(tmp_path)
        apd = _apd_csv(tmp_path)
        apd.write_text(apd.read_text() + "camelyon,Robust-A,-0.05,0.0\n")

        with pytest.raises(CaptionClaimError, match=r"(?s)apd\.csv.*finetune_pairs\.py"):
            basis.load(path, metadata=_metadata(), root=tmp_path)

    def test_a_missing_input_fails_with_its_remedy(self, tmp_path):
        path = _current_study(tmp_path)
        _metrics_csv(tmp_path, "pathorob-camelyon").unlink()

        with pytest.raises(CaptionClaimError, match=r"(?s)metrics\.csv.*finetune_pairs\.py"):
            basis.load(path, metadata=_metadata(), root=tmp_path)

    def test_a_study_that_records_no_input_fingerprints_fails(self, tmp_path):
        path = _current_study(tmp_path)
        pd.read_csv(path).drop(columns=FINGERPRINT_COLUMNS).to_csv(path, index=False)

        with pytest.raises(CaptionClaimError, match="finetune_pairs.py"):
            basis.load(path, metadata=_metadata(), root=tmp_path)

    def test_the_fingerprinted_inputs_are_the_manifest_run_and_the_apd_table(self):
        """The per-sample file is deliberately not fingerprinted: it is large, and the study
        already refuses per-sample points that do not reproduce ``metrics.csv``."""
        for benchmark in study.BENCHMARKS:
            assert study.input_rels(benchmark) == {
                "metrics_sha256": by_benchmark(benchmark).metrics_rel,
                "apd_sha256": "output/studies/apd/apd.csv",
            }


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

    def test_a_cohort_in_only_the_parents_pretraining_provenance_is_daggered_and_named(self):
        metadata = _metadata()
        metadata.loc[metadata["model"] == "Base-A", "corpus_domains"] = "tcga"

        tex = _render(metadata=metadata)

        assert r"TCGA-4x4$^{\dagger}$" in _block(tex, "Robust-A")
        assert "TCGA-4x4 for Base-A" in _caption(tex)

    def test_without_exposure_no_dagger_is_drawn(self):
        assert r"\dagger" not in _render().replace(r"\ddagger", "")
