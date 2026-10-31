"""Builder tests for ``tab:silhouette-baselines`` and its macros (``scripts/repro``).

The renderer and the macro family run on synthetic study artifacts written to a temporary
directory, so no ``paper/`` or ``output/`` tree is needed. The caption's numbers come from
the study's own provenance and tables; the tests change those inputs and watch the caption
follow, and break the claims it makes (the control is excluded, CRoMa's row is the one
``tab:apd-correlation`` reports, the q values adjust the stated family) and watch it raise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "repro", ROOT / "scripts" / "studies"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import generate_paper_values as gpv  # noqa: E402
import generate_silhouette_baselines_table as gsb  # noqa: E402
import silhouette_baselines as study  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

COHORTS = ("camelyon", "tcga_4x4", "tolkach")
N_RANKED = 10


def _per_encoder() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for dataset in COHORTS:
        latent = rng.normal(size=N_RANKED + 1)
        models = [f"enc-{i}" for i in range(N_RANKED)] + [study.CONTROL_MODEL]
        frame = pd.DataFrame(
            {
                "dataset": dataset,
                "model": models,
                "ranked": [m != study.CONTROL_MODEL for m in models],
                "croma": latent + rng.normal(scale=0.4, size=N_RANKED + 1),
                "nipd_id": latent + rng.normal(scale=0.6, size=N_RANKED + 1),
            }
        )
        for column in study.SCORE_COLUMNS:
            frame[column] = latent + rng.normal(size=N_RANKED + 1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _provenance(**overrides) -> dict:
    provenance = {
        "implementation": "scib-metrics",
        "implementation_version": "0.5.10",
        "n_neighbors": 90,
        "perplexity": 30,
        "inference": {"n_boot": 200, "n_permutations": 300, "level": 0.95, "seed": 0},
    }
    provenance.update(overrides)
    return provenance


def _apd_correlation(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of tab:apd-correlation's source that the CRoMa cross-check reads."""
    croma = table[(table["kind"] == "downstream") & (table["metric"] == "croma")]
    return pd.DataFrame(
        {
            "target": "nipd_id",
            "metric": "croma",
            "scope": croma["dataset"].to_numpy(),
            "n": croma["n"].to_numpy(),
            "spearman": croma["estimate"].to_numpy(),
        }
    )


def _write(
    tmp_path: Path,
    *,
    per_encoder: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
    provenance: dict | None = None,
    apd: pd.DataFrame | None = None,
) -> dict[str, Path]:
    per_encoder = _per_encoder() if per_encoder is None else per_encoder
    if table is None:
        table = study.associations(per_encoder, n_boot=200, n_permutations=300)
    paths = {
        "per_encoder_csv": tmp_path / "silhouette_baselines.csv",
        "correlation_csv": tmp_path / "silhouette_baselines_correlation.csv",
        "provenance_json": tmp_path / "silhouette_baselines.json",
        "apd_correlation_csv": tmp_path / "apd_correlation.csv",
    }
    per_encoder.to_csv(paths["per_encoder_csv"], index=False)
    table.to_csv(paths["correlation_csv"], index=False)
    paths["provenance_json"].write_text(json.dumps(provenance or _provenance()))
    (_apd_correlation(table) if apd is None else apd).to_csv(
        paths["apd_correlation_csv"], index=False
    )
    return paths


def _caption(tex: str) -> str:
    return tex[tex.index(r"\caption{") :]


class TestTable:
    def test_each_cohort_reports_agreement_downstream_and_head_to_head_rows(
        self, tmp_path: Path
    ) -> None:
        tex = gsb.build_table(gsb.load(**_write(tmp_path)))
        assert r"\label{tab:silhouette-baselines}" in tex
        body = tex[: tex.index(r"\caption{")]
        for cohort in ("Camelyon", "TCGA-4x4", "Tolkach-ESCA"):
            assert rf"\multicolumn{{6}}{{l}}{{\textbf{{{cohort}}}}}" in body
        rows = [line for line in body.splitlines() if line.rstrip().endswith(r"\\")]
        metric_rows = [r for r in rows if "multicolumn" not in r and "Estimate" not in r]
        assert len(metric_rows) == 3 * 7
        # agreement and head-to-head rows carry an interval only; downstream rows p and q
        with_p = [r for r in metric_rows if r.count("--") == 0]
        assert len(with_p) == 3 * 3
        assert all(r.count("--") == 2 for r in metric_rows if r not in with_p)

    def test_cells_print_the_study_estimates_and_intervals(self, tmp_path: Path) -> None:
        paths = _write(tmp_path)
        table = pd.read_csv(paths["correlation_csv"])
        row = table[
            (table["dataset"] == "tolkach")
            & (table["kind"] == "head_to_head")
            & (table["metric"] == "ilisi")
        ].iloc[0]
        tex = gsb.build_table(gsb.load(**paths))
        assert rf"${row['estimate']:.2f}$ & $[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]$" in tex

    def test_caption_reads_its_settings_from_the_study(self, tmp_path: Path) -> None:
        tex = gsb.build_table(gsb.load(**_write(tmp_path)))
        caption = _caption(tex)
        for fragment in ("v0.5.10", "k{=}90", "perplexity~$30$", "L2-normalised", "$10$ ranked"):
            assert fragment in caption
        assert "Benjamini--Hochberg" in caption and "$9$ downstream" in caption
        assert "own slide or case" in caption
        moved = _provenance(implementation_version="0.6.0", n_neighbors=60, perplexity=20)
        caption = _caption(gsb.build_table(gsb.load(**_write(tmp_path, provenance=moved))))
        assert "v0.6.0" in caption and "k{=}60" in caption and "perplexity~$20$" in caption

    def test_caption_counts_the_ranked_encoders_tied_on_the_ilisi_floor(
        self, tmp_path: Path
    ) -> None:
        eps = float(np.finfo(np.float32).eps)
        per_encoder = _per_encoder()
        per_encoder["ilisi"] = 0.05 + per_encoder["ilisi"].abs()
        floor = {"camelyon": [0.0, eps, 4 * eps], "tolkach": [-eps]}
        for dataset, values in floor.items():
            rows = per_encoder.index[(per_encoder["dataset"] == dataset) & per_encoder["ranked"]]
            per_encoder.loc[rows[: len(values)], "ilisi"] = values
        # the control on the floor is not one of the ranked encoders the caption counts
        per_encoder.loc[per_encoder["model"] == study.CONTROL_MODEL, "ilisi"] = 0.0

        caption = _caption(gsb.build_table(gsb.load(**_write(tmp_path, per_encoder=per_encoder))))

        assert "$3$, $0$ and $1$ of the $10$" in caption
        assert "Camelyon, TCGA-4x4 and Tolkach-ESCA" in caption

    def test_caption_gives_ilisi_ranked_as_returned_where_the_floor_tie_moves_a_rank(
        self, tmp_path: Path
    ) -> None:
        # Camelyon's four floor values are ordered by rounding as returned, so tying them
        # moves ranks; Tolkach-ESCA's single floor value moves none. The caption gives the
        # untied rho and delta for Camelyon only, from the study's *_as_returned rows.
        eps = float(np.finfo(np.float32).eps)
        per_encoder = _per_encoder()
        per_encoder["ilisi"] = 0.05 + per_encoder["ilisi"].abs()
        floor = {"camelyon": [3 * eps, -eps, 0.0, 2 * eps], "tolkach": [eps]}
        for dataset, values in floor.items():
            rows = per_encoder.index[(per_encoder["dataset"] == dataset) & per_encoder["ranked"]]
            per_encoder.loc[rows[: len(values)], "ilisi"] = values
        paths = _write(tmp_path, per_encoder=per_encoder)
        table = pd.read_csv(paths["correlation_csv"])

        def cell(kind: str) -> str:
            row = table[
                (table["dataset"] == "camelyon")
                & (table["kind"] == kind)
                & (table["metric"] == "ilisi")
            ].iloc[0]
            text = f"${row['estimate']:.2f}$ $[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]$"
            return text.replace("-0.00", "0.00")

        caption = _caption(gsb.build_table(gsb.load(**paths)))

        sentence = caption[caption.index("Ranked as returned") :]
        assert (
            f"{cell('downstream_as_returned')} and {cell('head_to_head_as_returned')}" in sentence
        )
        assert cell("downstream_as_returned") != cell("downstream")
        assert "on Camelyon;" in sentence and "Tolkach" not in sentence
        assert sentence.startswith("Ranked as returned, without the tie,")
        assert "on Camelyon; elsewhere the tie moves no rank." in sentence
        assert len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", caption)) <= 350

    def test_caption_says_so_when_the_floor_tie_moves_no_rank(self, tmp_path: Path) -> None:
        caption = _caption(gsb.build_table(gsb.load(**_write(tmp_path))))
        assert "iLISI's rows are the same: the tie moves no rank." in caption
        assert "would be" not in caption and "elsewhere" not in caption

    def test_caption_is_within_the_word_limit(self, tmp_path: Path) -> None:
        # Counted as the paper's own legend check counts: every alphanumeric run, LaTeX
        # command names included, against the journal's 350-word cap.
        caption = _caption(gsb.build_table(gsb.load(**_write(tmp_path))))
        assert len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", caption)) <= 350


class TestClaims:
    def test_croma_disagreeing_with_tab_apd_correlation_raises(self, tmp_path: Path) -> None:
        per_encoder = _per_encoder()
        table = study.associations(per_encoder, n_boot=200, n_permutations=300)
        apd = _apd_correlation(table)
        apd.loc[apd["scope"] == "tcga_4x4", "spearman"] += 0.01
        baselines = gsb.load(**_write(tmp_path, per_encoder=per_encoder, table=table, apd=apd))
        with pytest.raises(CaptionClaimError, match="tcga_4x4"):
            gsb.build_table(baselines)

    def test_ranking_over_the_full_panel_raises(self, tmp_path: Path) -> None:
        per_encoder = _per_encoder()
        table = study.associations(per_encoder.assign(ranked=True), n_boot=200, n_permutations=300)
        baselines = gsb.load(**_write(tmp_path, per_encoder=per_encoder, table=table))
        with pytest.raises(CaptionClaimError, match="control"):
            gsb.build_table(baselines)

    def test_q_values_outside_the_stated_family_raise(self, tmp_path: Path) -> None:
        per_encoder = _per_encoder()
        table = study.associations(per_encoder, n_boot=200, n_permutations=300)
        table.loc[table["kind"] == "downstream", "q_value"] = table["p_value"]
        baselines = gsb.load(**_write(tmp_path, per_encoder=per_encoder, table=table))
        with pytest.raises(CaptionClaimError, match="Benjamini"):
            gsb.build_table(baselines)

    def test_a_table_without_the_untied_ilisi_rows_fails(self, tmp_path: Path) -> None:
        per_encoder = _per_encoder()
        table = study.associations(per_encoder, n_boot=200, n_permutations=300)
        table = table[~table["kind"].str.endswith("_as_returned")]
        baselines = gsb.load(**_write(tmp_path, per_encoder=per_encoder, table=table))
        with pytest.raises(KeyError, match="as_returned"):
            gsb.build_table(baselines)

    def test_untied_rows_that_disagree_with_an_unmoved_ranking_raise(self, tmp_path: Path) -> None:
        # No floor value moves a rank here, so the untied rows must be the reported ones.
        per_encoder = _per_encoder()
        table = study.associations(per_encoder, n_boot=200, n_permutations=300)
        stale = (table["dataset"] == "tolkach") & (table["kind"] == "downstream_as_returned")
        table.loc[stale, "estimate"] += 0.01
        baselines = gsb.load(**_write(tmp_path, per_encoder=per_encoder, table=table))
        with pytest.raises(CaptionClaimError, match="tolkach"):
            gsb.build_table(baselines)

    def test_a_missing_study_artifact_fails_rather_than_skips(self, tmp_path: Path) -> None:
        paths = _write(tmp_path)
        paths["correlation_csv"].unlink()
        with pytest.raises(FileNotFoundError, match="silhouette_baselines.py"):
            gsb.load(**paths)


def test_build_paper_fails_rather_than_skips_without_the_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import build_paper

    ran: list[str] = []
    monkeypatch.setattr(build_paper, "REPO", tmp_path)  # no study artifacts under it
    monkeypatch.setattr(build_paper, "_run", lambda script, extra: ran.append(script) or 0)
    monkeypatch.setattr(sys, "argv", ["build_paper.py"])

    assert build_paper.main() == 1
    out = capsys.readouterr().out
    failed = [line for line in out.splitlines() if line.startswith("FAILED")]
    assert failed and "generate_silhouette_baselines_table.py" in failed[0]
    assert "SKIPPED  generate_silhouette_baselines_table.py" not in out
    assert "generate_silhouette_baselines_table.py" not in ran


def test_the_one_command_reproduction_runs_the_study_after_the_apd_steps_it_reads() -> None:
    import reproduce_faithful

    scripts = [script for _, script, _ in reproduce_faithful.STUDIES]
    study_script = "scripts/studies/silhouette_baselines.py"
    assert study_script in scripts
    # it joins apd_metrics_joined.csv and is checked against apd_correlation.csv
    for upstream in ("apd_experiment.py", "apd_croma_correlation.py"):
        (position,) = [i for i, s in enumerate(scripts) if s.endswith(upstream)]
        assert scripts.index(study_script) > position


class TestMacros:
    def test_downstream_rhos_and_head_to_head_differences_carry_ci_bounds(
        self, tmp_path: Path
    ) -> None:
        paths = _write(tmp_path)
        table = pd.read_csv(paths["correlation_csv"])
        lines = gpv._silhouette_baseline_macros(gsb.load(**paths))
        macros = dict(
            re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line).groups() for line in lines
        )

        def value(dataset: str, kind: str, metric: str, column: str) -> str:
            row = table[
                (table["dataset"] == dataset)
                & (table["kind"] == kind)
                & (table["metric"] == metric)
            ].iloc[0]
            return f"{row[column]:.2f}".replace("-0.00", "0.00")

        for dataset, scope in (
            ("camelyon", "Camelyon"),
            ("tcga_4x4", "Tcga"),
            ("tolkach", "Tolkach"),
        ):
            for metric, name in (("croma", "Croma"), ("batch_asw", "BatchAsw"), ("ilisi", "Ilisi")):
                for column, suffix in (("estimate", ""), ("ci_lo", "Lo"), ("ci_hi", "Hi")):
                    assert macros[f"BaselinesNipdId{name}{scope}{suffix}"] == value(
                        dataset, "downstream", metric, column
                    )
            for metric, name in (("batch_asw", "BatchAsw"), ("ilisi", "Ilisi")):
                for column, suffix in (("estimate", ""), ("ci_lo", "Lo"), ("ci_hi", "Hi")):
                    assert macros[f"BaselinesDelta{name}{scope}{suffix}"] == value(
                        dataset, "head_to_head", metric, column
                    )
        assert len(macros) == len(lines) == 3 * (3 + 2) * 3
