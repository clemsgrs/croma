"""The slide-vs-meanpool study's paper surfaces: ``tab:slide-vs-meanpool`` and its macros.

Also the parts of the study that the paper's tooling drives: the paper values, which carry
the paired differences, and the full reproduction's study commands. The sdist leaves this
module out together with ``scripts/repro``; the rest of the study is tested in
``test_slide_vs_meanpool.py``, which it ships.

Everything renders from synthetic study artifacts in a ``tmp_path`` tree; no ``paper/`` or
``output/`` tree is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts" / "studies" / "apd", ROOT / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# --- the paper table ----------------------------------------------------------------

REPRO = ROOT / "scripts" / "repro"


def _repro():
    if str(REPRO) not in sys.path:
        sys.path.insert(0, str(REPRO))
    import _slide_vs_meanpool as basis
    import generate_slide_vs_meanpool_table as table

    return basis, table


def _write_study(study_dir: Path, pairs=None, not_run=None) -> Path:
    """Study artifacts as ``evaluate`` writes them, with hand-picked values per pair.

    ``pairs`` defaults to the rendered pairs, and ``not_run`` (the pairs the study records
    as not run, with their reasons) to the study's own declaration.
    """
    import json

    import slide_vs_meanpool as study

    basis, _ = _repro()
    pairs = basis.RENDERED_PAIRS if pairs is None else pairs
    not_run = study.NOT_RUN if not_run is None else not_run
    study_dir.mkdir(parents=True, exist_ok=True)
    summary, differences, reproduction = [], [], {}
    for i, pair in enumerate(pairs):
        reproduced = pair != "TITAN"
        slide_arm = pair if reproduced else f"{pair}-rerun"
        for role, model, values in (
            ("slide", slide_arm, (-0.304, 0.895, -0.6017, -0.1886, -0.1258)),
            ("meanpool", f"{pair}-meanpool", (-0.204, 0.795, -0.5017, -0.0886, -0.0258)),
        ):
            croma, f0, ltm, nipd_id, nipd_ood = values
            summary.append(
                dict(
                    pair=pair,
                    arm=role,
                    model=model,
                    bio_knn_bacc=[0.915, 0.971, 0.975, 0.962, 0.958][i],
                    croma=croma,
                    croma_f0=f0,
                    croma_ltm_alpha=ltm,
                    nipd_id=nipd_id,
                    nipd_ood=nipd_ood,
                )
            )
        for statistic, point in (("median_croma", -0.1), ("f0", 0.2), ("ltm", -0.1)):
            differences.append(
                dict(
                    pair=pair,
                    statistic=statistic,
                    point=point,
                    lo=point - 0.05,
                    hi=point + 0.04,
                    spread=np.nan,
                    n=2000,
                )
            )
        for statistic in ("nipd_id", "nipd_ood"):
            differences.append(
                dict(
                    pair=pair,
                    statistic=statistic,
                    point=-0.1,
                    lo=np.nan,
                    hi=np.nan,
                    spread=0.012,
                    n=20,
                )
            )
        reproduction[pair] = dict(
            slide_arm=slide_arm,
            reproduced=reproduced,
            n_slides=1324,
            n_reproduced=1324 if reproduced else 17,
            min_cosine=0.99991 if reproduced else 0.4123,
        )
    pd.DataFrame(summary).to_csv(study_dir / "summary.csv", index=False)
    pd.DataFrame(differences).to_csv(study_dir / "differences.csv", index=False)
    provenance = {
        "pairs": {p: {"reproduction": reproduction[p]} for p in pairs},
        "not_run": dict(not_run),
        "bootstrap": {"n_boot": 2000, "level": 0.95},
        "downstream": {"iterations": 20},
        "representation": {"protocol": "k-star", "croma_headline_m": 5},
    }
    (study_dir / "provenance.json").write_text(json.dumps(provenance))
    return study_dir


def test_the_table_renders_every_declared_pair_with_both_arms_and_paired_cis(tmp_path):
    basis, table = _repro()

    tex = table.build(basis.load(_write_study(tmp_path / "study")))

    assert r"\label{tab:slide-vs-meanpool}" in tex
    for pair in basis.RENDERED_PAIRS:
        assert rf"\multirow{{3}}{{*}}{{{pair}}}" in tex
    # The TITAN block: its re-run slide arm, the mean-pooled arm, then slide - mean-pool.
    assert r"& slide & $0.915$ & $-0.30$ & $0.895$ & $-0.60$ & $-18.9$ & $-12.6$ \\" in tex
    assert r"& mean-pool & $0.915$ & $-0.20$ & $0.795$ & $-0.50$ & $-8.9$ & $-2.6$ \\" in tex
    assert (
        r"& $\Delta$ & & $-0.10$\,{\scriptsize$[-0.15, -0.06]$} "
        r"& $0.200$\,{\scriptsize$[0.150, 0.240]$} & $-0.10$\,{\scriptsize$[-0.15, -0.06]$} "
        r"& $-10.0$\,{\scriptsize$\pm1.2$} & $-10.0$\,{\scriptsize$\pm1.2$} \\"
    ) in tex


def test_the_caption_names_rendered_and_missing_pairs_and_the_reproduction_result(tmp_path):
    basis, table = _repro()
    import slide_vs_meanpool as study

    tex = table.build(basis.load(_write_study(tmp_path / "study")))
    caption = tex[tex.index(r"\caption{") :]

    declared = [pair.slide_model for pair in study.PAIRS]
    missing = [p for p in declared if p not in basis.RENDERED_PAIRS]
    assert missing, "every declared pair is rendered, so the missing-pair clause asserts nothing"
    for pair in [*basis.RENDERED_PAIRS, *missing]:
        assert pair in caption
    assert "mean of the same" in caption  # how the mean-pooled arm is built
    # TITAN's re-run did not reproduce, so its re-run slide embeddings are the slide arm.
    assert "TITAN: $17$ of $1{,}324$ slides" in caption
    assert "minimum cosine $0.412$" in caption
    # Every pair left out is one the study did not run, named with the reason it recorded.
    assert missing == list(study.NOT_RUN)
    for pair in missing:
        assert (
            f"{pair} ({basis.tile_display(pair)}), an optional pair, was not run: "
            f"{study.NOT_RUN[pair]}." in caption
        )
    assert "Declared pairs not shown" not in caption


def test_every_declared_pair_is_either_rendered_or_recorded_as_not_run(tmp_path):
    # A declared pair is never dropped silently: left out of the table, it must be one the
    # study recorded as not run, with a reason.
    basis, _ = _repro()

    with pytest.raises(basis.CaptionClaimError, match="Prov-GigaPath.*neither rendered"):
        basis.load(_write_study(tmp_path / "no-record", not_run={}))
    with pytest.raises(basis.CaptionClaimError, match="Prov-GigaPath.*neither rendered"):
        basis.load(_write_study(tmp_path / "no-reason", not_run={"Prov-GigaPath": " "}))


def test_a_pair_the_study_did_not_run_is_never_rendered(tmp_path):
    basis, _ = _repro()
    rendered = (*basis.RENDERED_PAIRS, "Prov-GigaPath")
    study_dir = _write_study(tmp_path / "study", pairs=rendered)

    with pytest.raises(basis.CaptionClaimError, match="Prov-GigaPath.*not run"):
        basis.load(study_dir, rendered=rendered)


def test_a_required_pair_is_never_left_out_even_if_recorded_as_not_run(tmp_path):
    import slide_vs_meanpool as study

    basis, _ = _repro()
    rendered = tuple(p for p in basis.RENDERED_PAIRS if p != "TITAN")
    not_run = {**study.NOT_RUN, "TITAN": "the GPU was busy"}
    study_dir = _write_study(tmp_path / "study", pairs=rendered, not_run=not_run)

    with pytest.raises(basis.CaptionClaimError, match="TITAN.*required"):
        basis.load(study_dir, rendered=rendered)


def test_every_pair_that_ran_is_rendered():
    # The four pairs the re-extraction finished are shown; only the not-run one is not.
    import slide_vs_meanpool as study

    basis, _ = _repro()
    assert basis.RENDERED_PAIRS == ("TITAN", "MOOZY", "PRISM2", "PRISM")
    declared = [pair.slide_model for pair in study.PAIRS]
    assert sorted([*basis.RENDERED_PAIRS, *study.NOT_RUN]) == sorted(declared)


def test_a_not_run_reason_is_set_as_text_in_the_caption(tmp_path):
    import slide_vs_meanpool as study

    basis, table = _repro()
    reason = r"50% of runs_failed & {1} #2 ~x ^y \z"
    study_dir = _write_study(tmp_path / "study", not_run={"Prov-GigaPath": reason})

    caption = _caption(table.build(basis.load(study_dir)))

    assert (
        r"was not run: 50\% of runs\_failed \& \{1\} \#2 \textasciitilde{}x "
        r"\textasciicircum{}y \textbackslash{}z." in caption
    )
    assert study.NOT_RUN["Prov-GigaPath"] == basis.tex_text(study.NOT_RUN["Prov-GigaPath"])


#: The table ``_write_study`` renders, as it rendered before the caption counted the
#: replicates each nIPD spread is over. With every replicate defined it must not move. (It
#: last changed when PRISM, which ran, joined the rendered pairs and the caption began to
#: name the pair not run, with its reason; the three required pairs' rows did not change.)
GOLDEN_TABLE = ROOT / "tests" / "fixtures" / "slide_vs_meanpool_table.tex"


def _defined_replicates(study_dir: Path, pair: str, statistic: str, n: int, spread: float):
    """Rewrite one nIPD difference as if both arms' nIPD were defined on ``n`` replicates."""
    path = study_dir / "differences.csv"
    frame = pd.read_csv(path)
    row = (frame["pair"] == pair) & (frame["statistic"] == statistic)
    frame.loc[row, "n"] = n
    frame.loc[row, "spread"] = spread
    frame.to_csv(path, index=False)


def _caption(tex: str) -> str:
    return tex[tex.index(r"\caption{") :]


def test_with_every_replicate_defined_the_table_is_unchanged(tmp_path):
    basis, table = _repro()

    tex = table.build(basis.load(_write_study(tmp_path / "study")))

    assert tex == GOLDEN_TABLE.read_text()
    assert r"($\pm$ the standard deviation of the $20$ paired probe replicates)" in tex


def test_an_undefined_nipd_spread_renders_a_dash_never_nan(tmp_path):
    basis, table = _repro()
    study_dir = _write_study(tmp_path / "study")
    _defined_replicates(study_dir, "MOOZY", "nipd_ood", n=1, spread=np.nan)

    tex = table.build(basis.load(study_dir))
    macros = basis.macros(basis.load(study_dir))

    assert "nan" not in tex.lower()
    assert r"& $-10.0$\,{\scriptsize$\pm1.2$} & $-10.0$\,{\scriptsize$\pm$---} \\" in tex
    assert "; --- where fewer than two are)" in _caption(tex)
    assert r"\newcommand{\SlideMeanpoolMoozyNipdOodDiffSd}{---}" in macros
    assert r"\newcommand{\SlideMeanpoolMoozyNipdIdDiffSd}{$1.2$}" in macros
    assert not any("nan" in line.lower() for line in macros)


def test_the_caption_counts_only_the_replicates_both_arms_define(tmp_path):
    basis, table = _repro()
    study_dir = _write_study(tmp_path / "study")
    rule = r"paired probe replicates in which both arms' \code{nIPD} is defined"

    # One replicate drops out of every comparison: the caption gives the count it keeps.
    for pair in basis.RENDERED_PAIRS:
        for statistic in ("nipd_id", "nipd_ood"):
            _defined_replicates(study_dir, pair, statistic, n=19, spread=0.012)
    caption = _caption(table.build(basis.load(study_dir)))
    assert rf"the standard deviation of the $19$ of $20$ {rule})" in caption

    # Counts that differ by comparison: the rule, and no count that some row lacks.
    _defined_replicates(study_dir, "MOOZY", "nipd_ood", n=18, spread=0.012)
    caption = _caption(table.build(basis.load(study_dir)))
    assert rf"the standard deviation over the {rule})" in caption
    assert "$19$" not in caption and "$20$" not in caption


def test_a_declared_pair_without_study_output_fails_the_build(tmp_path):
    basis, _ = _repro()
    study_dir = _write_study(tmp_path / "study", pairs=("TITAN", "MOOZY"))

    with pytest.raises(basis.CaptionClaimError, match="PRISM2"):
        basis.load(study_dir)


def test_the_table_and_macros_rebuild_byte_for_byte(tmp_path):
    basis, table = _repro()
    study_dir = _write_study(tmp_path / "study")
    out = tmp_path / "slide_vs_meanpool.tex"

    table.main(["--study-dir", str(study_dir), "--out", str(out)])
    first = out.read_bytes()
    table.main(["--study-dir", str(study_dir), "--out", str(out)])
    assert out.read_bytes() == first

    macros = basis.macros(basis.load(study_dir))
    assert macros == basis.macros(basis.load(study_dir))
    assert r"\newcommand{\SlideMeanpoolTitanCromaDiff}{$-0.10$}" in macros
    assert r"\newcommand{\SlideMeanpoolPrismTwoNipdOodDiffSd}{$1.2$}" in macros
    assert r"\newcommand{\SlideMeanpoolMoozyFzeroDiffCI}{$[0.150, 0.240]$}" in macros


def test_the_paper_values_carry_the_paired_differences_or_fail_on_a_missing_pair(tmp_path):
    basis, _ = _repro()
    import generate_paper_values as gpv

    # Not run at all: the study is skipped by name, like every other study block.
    assert gpv._slide_vs_meanpool_macros(tmp_path) == []

    study_dir = _write_study(tmp_path / basis.STUDY_REL)
    lines = gpv._slide_vs_meanpool_macros(tmp_path)
    assert lines == basis.macros(basis.load(study_dir))

    _write_study(tmp_path / basis.STUDY_REL, pairs=("TITAN", "MOOZY"))
    with pytest.raises(basis.CaptionClaimError, match="PRISM2"):
        gpv._slide_vs_meanpool_macros(tmp_path)


# --- the one-command reproduction ------------------------------------------------------


def test_the_full_reproduction_reduces_then_evaluates_the_study():
    # The studies run in list order, and build_paper.py runs after all of them.
    _repro()
    import reproduce_faithful

    commands = [" ".join([script, *extra]) for _, script, extra in reproduce_faithful.STUDIES]

    def position(needle: str) -> int:
        return next(i for i, command in enumerate(commands) if needle in command)

    assert position("slide_vs_meanpool.py reduce") < position(
        "slide_vs_meanpool.py evaluate --recompute"
    )
    # Both steps take the study-root default for the arms, so evaluate reads what reduce
    # wrote and neither writes beside the published embeddings.
    study_commands = [c for c in commands if "slide_vs_meanpool.py" in c]
    assert len(study_commands) == 2
    assert not any("--embeddings-dir" in c for c in study_commands)
    # Both steps take exactly the pairs the table renders, which are more than the
    # study's default (the required pairs): one ran optionally.
    basis, _ = _repro()
    for _, script, extra in reproduce_faithful.STUDIES:
        if "slide_vs_meanpool.py" in script:
            start = extra.index("--pairs") + 1
            assert extra[start : start + len(basis.RENDERED_PAIRS)] == list(basis.RENDERED_PAIRS)
            assert extra[start + len(basis.RENDERED_PAIRS) :] == []


def test_the_reproduction_runs_the_evaluation_with_its_own_worker_count_even_serially():
    # The evaluation's own default is 3 APD workers, so a reproduction asked for one worker
    # must say so. One rule covers every study that takes --jobs: it is always told the
    # count, 1 included (the probe sweeps default to 1, so for them that changes nothing).
    _repro()
    import reproduce_faithful

    studies = {label: (label, script, extra) for label, script, extra in reproduce_faithful.STUDIES}

    def command(label: str, jobs: int) -> list[str]:
        return reproduce_faithful.study_command(*studies[label], jobs=jobs)

    assert "slide-vs-meanpool-evaluate" in reproduce_faithful.JOBS_STUDIES
    evaluate = command("slide-vs-meanpool-evaluate", jobs=1)
    assert evaluate[-2:] == ["--jobs", "1"]
    assert "--recompute" in evaluate
    assert command("slide-vs-meanpool-evaluate", jobs=4)[-2:] == ["--jobs", "4"]
    for sweep in ("apd-sweep", "auc-sweep"):
        assert command(sweep, jobs=1)[-2:] == ["--jobs", "1"]
        assert command(sweep, jobs=4)[-2:] == ["--jobs", "4"]
    assert "--jobs" not in command("slide-vs-meanpool-reduce", jobs=4)
    assert "--jobs" not in command("auc-report", jobs=4)
