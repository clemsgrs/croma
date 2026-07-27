"""The paper's results tables are build outputs, and this is what enforces it (ADR-0010).

Two kinds of test live here. The *builder* tests drive the pure renderer on synthetic runs
and pin the behaviour that hand-editing eroded: the control band, the bold rule, and
descriptive-only captions. The *freshness* test re-renders each manifest entry and fails if the
``.tex`` on disk differs -- the check that would have caught a table sitting five models and
one protocol behind its own ``metrics.csv``.

``paper/`` is git-ignored, so the freshness test skips in a fresh checkout. It is the only
place the drift is visible at all, since ``git status`` never shows it.

**This means the freshness test never runs in CI** -- every one of its cases skips there, and
a green CI run says nothing about whether the paper is fresh. It is a local pre-flight, and
it only protects the manuscript if it is run deliberately::

    PYTHONPATH=src python -m pytest tests/test_paper_artifacts.py

Run it before any paper build, and treat a skip on a machine that *has* ``paper/`` as a
failure to configure, not as a pass. The builder tests below do run in CI; they pin the
renderer's behaviour, not the artifacts' freshness.

Prior art: tests/test_generate_model_tables.py.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
for _p in (str(ROOT / "src"), str(REPRO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _cross_benchmark as _cb  # noqa: E402
import _distributions as _dist  # noqa: E402
import _rank_pareto as _rp  # noqa: E402
import generate_cross_benchmark_float as gcb  # noqa: E402
import generate_distribution_floats as gdf  # noqa: E402
import generate_pareto_float as gpf  # noqa: E402
import generate_rank_pareto_float as grp  # noqa: E402
import generate_m_sweep_table as gms  # noqa: E402
import generate_panda_table as gpt  # noqa: E402
import generate_paper_values as gpv  # noqa: E402
import generate_results_table as grt  # noqa: E402
import paper_manifest as pm  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402
from croma.plotstyle import CONTROL_MODEL  # noqa: E402

HEADLINE_M = int(grt.CROMA_HEADLINE_M)


def _write_run(tmp_path: Path, rows: list[dict], k: int | list[int] = 11) -> Path:
    """Materialise a minimal ``results/`` dir: metrics.csv + per_sample_metrics.csv."""
    ks = [k] * len(rows) if isinstance(k, int) else k
    frame = pd.DataFrame([
        {
            "model": r["model"],
            "k": ks[i],
            "bio_knn_bacc": r.get("bio", 0.95),
            "confounder_knn_bacc": r.get("conf", 0.95),
            "ri": r.get("ri", 0.5),
            "mari": r.get("mari", 0.5),
            "croma": r["croma"],
            "croma_ltm_alpha": r.get("ltm", -0.2),
            "ri_undefined_frac": 1.0 - r.get("support", 0.3),
            "confounder_display_name": "Medical Center",
        }
        for i, r in enumerate(rows)
    ])
    out = tmp_path / "results"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "metrics.csv", index=False)
    # One per-sample row per model is enough to define F(0) = fraction with CRoMa < 0.
    pd.DataFrame({
        "model": frame["model"],
        f"croma_m{HEADLINE_M}": frame["croma"],
    }).to_csv(out / "per_sample_metrics.csv", index=False)
    return out / "metrics.csv"


def _entry(**kw) -> pm.ResultsTable:
    base = dict(prefix="T", benchmark="b", protocol="median-k",
                display_name="Test", label="tab:test")
    return pm.ResultsTable(**{**base, **kw})


def _caption_line(tex: str) -> str:
    """Return the generated one-line legend, excluding result rows and plot metadata."""
    return next(line for line in tex.splitlines() if r"\caption{" in line)


# A healthy run: two encoders plus the control, which has the most support and the worst
# biology -- the shape every real tile benchmark has.
HEALTHY = [
    {"model": "Good", "croma": 0.20, "conf": 0.95, "support": 0.40, "bio": 0.98, "ri": 0.8, "mari": 0.82},
    {"model": "Bad", "croma": -0.30, "conf": 1.000, "support": 0.20, "bio": 0.96, "ri": 0.02, "mari": 0.01},
    {"model": CONTROL_MODEL, "croma": 0.05, "conf": 0.91, "support": 0.68, "bio": 0.90},
]


class TestControlBand:
    def test_control_is_rendered_below_a_rule_and_last(self, tmp_path):
        tex = grt.build_table(_entry(), _write_run(tmp_path, HEALTHY))
        body = tex.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]
        rows = [ln for ln in body.splitlines() if ln.endswith(r"\\")]
        assert rows[-1].startswith(CONTROL_MODEL)
        # ranked rows, rule, control row, closing rule
        assert body.count(r"\hline") == 4

    def test_control_never_wins_a_bold_even_when_it_leads_a_column(self, tmp_path):
        """The control holds the highest support (68%) and must still not be bolded."""
        tex = grt.build_table(_entry(), _write_run(tmp_path, HEALTHY))
        control_row = [ln for ln in tex.splitlines() if ln.startswith(CONTROL_MODEL)][0]
        assert r"\textbf" not in control_row
        # ...and the bold went to the best *ranked* model instead.
        good_row = [ln for ln in tex.splitlines() if ln.startswith("Good")][0]
        assert r"\textbf{40.0\%}" in good_row

    def test_a_panel_without_the_control_renders_no_band(self, tmp_path):
        tex = grt.build_table(_entry(), _write_run(tmp_path, HEALTHY[:2]))
        assert CONTROL_MODEL not in tex
        assert tex.count(r"\hline") == 3
        assert "natural-image control" not in tex

    def test_diagnostic_columns_are_never_bolded(self, tmp_path):
        tex = grt.build_table(_entry(), _write_run(tmp_path, HEALTHY))
        # "Bad" holds the max conf bacc (1.000), which marks the *least* robust model.
        bad_row = [ln for ln in tex.splitlines() if ln.startswith("Bad")][0]
        assert r"\textbf{1.000}" not in bad_row


class TestCaptionClaims:
    """Legends describe the artifact structure without narrating the observed result."""

    def test_median_k_with_a_split_operating_point_raises(self, tmp_path):
        metrics = _write_run(tmp_path, HEALTHY, k=[9, 11, 11])
        with pytest.raises(grt.CaptionClaimError, match="shared operating point"):
            grt.build_table(_entry(protocol="median-k"), metrics)

    def test_k_star_describes_a_per_model_operating_point(self, tmp_path):
        tex = grt.build_table(_entry(protocol="k-star"), _write_run(tmp_path, HEALTHY, k=[3, 9, 9]))
        assert r"its own biological $k^\star$" in tex
        assert "shared operating point" not in tex

    def test_primary_caption_is_invariant_to_result_values(self, tmp_path):
        altered = [dict(r) for r in HEALTHY]
        altered[0].update(croma=-0.80, conf=0.51, support=0.99, bio=0.52)
        altered[1].update(croma=0.90, conf=0.52, support=0.98, bio=0.53)
        altered[2].update(croma=-0.70, conf=0.53, support=0.01, bio=0.99)

        original = grt.build_table(
            _entry(primary=True), _write_run(tmp_path / "original", HEALTHY)
        )
        changed = grt.build_table(
            _entry(primary=True), _write_run(tmp_path / "changed", altered)
        )
        assert _caption_line(original) == _caption_line(changed)
        assert "confounder-dominant" not in _caption_line(original)
        assert r"\code{Good}" not in _caption_line(original)

    def test_every_results_caption_defines_its_columns(self, tmp_path):
        metrics = _write_run(tmp_path, HEALTHY)
        assert "Columns: biological" in grt.build_table(_entry(primary=True), metrics)
        secondary = grt.build_table(_entry(primary=False), metrics)
        assert "Columns: biological" in secondary
        assert r"Table~\ref{tab:main-results}" not in secondary


class TestNotation:
    def test_tail_vocabulary_matches_the_glossary(self, tmp_path):
        """CONTEXT.md fixes F(0) and LTM_10; the retired forms must not reappear."""
        tex = grt.build_table(_entry(), _write_run(tmp_path, HEALTHY))
        assert "$F(0)$" in tex and r"$\mcode{LTM}_{10}$" in tex
        assert "P_{<0}" not in tex and r"LTM}_{10\%}" not in tex

    def test_float_placement_follows_the_paper_convention(self, tmp_path):
        tex = grt.build_table(_entry(), _write_run(tmp_path, HEALTHY))
        assert tex.startswith(r"\begin{table}[!htbp]")

    def test_ratio_scale_runs_are_normalised_to_margin(self, tmp_path):
        """A run stored as the raw ratio must print the margin, as the macros do."""
        rows = [{"model": "R", "croma": 1.5, "support": 0.3}, {"model": "S", "croma": 3.0, "support": 0.3}]
        tex = grt.build_table(_entry(), _write_run(tmp_path, rows))
        assert "0.20" in tex and "0.50" in tex  # (1.5-1)/(1.5+1), (3-1)/(3+1)
        assert "1.50" not in tex


class TestManifest:
    def test_exactly_one_primary_table(self):
        assert pm.primary().label == "tab:main-results"

    def test_prefixes_are_unique(self):
        prefixes = [t.prefix for t in pm.TABLES]
        assert len(prefixes) == len(set(prefixes))

    def test_rendered_entries_all_carry_an_output_path(self):
        assert all(t.out_tex and t.label for t in pm.rendered())

    def test_benchmarks_are_unique_so_by_benchmark_is_total(self):
        benchmarks = [t.benchmark for t in pm.TABLES]
        assert len(benchmarks) == len(set(benchmarks))
        for t in pm.TABLES:
            assert pm.by_benchmark(t.benchmark) is t

    def test_derived_paths_hang_off_the_run_directory(self):
        entry = pm.by_benchmark("pathorob-camelyon")
        assert entry.protocol == "median-k"  # the tile panel's reported protocol
        assert entry.run_rel == "output/metrics/median-k/pathorob-camelyon"
        assert entry.metrics_rel.startswith(entry.run_rel)
        assert entry.per_sample_rel.startswith(entry.run_rel)
        assert entry.studies_rel.startswith(entry.run_rel)


#: The scripts that build ``paper/sections/``. Every one of them once named a protocol
#: itself; all five said ``k-star``, and when the tile panel moved to ``median-k`` the old
#: runs were archived, so each silently read a directory that no longer existed.
_PAPER_PIPELINE = [
    REPRO / "generate_results_table.py",
    REPRO / "generate_panda_table.py",
    REPRO / "generate_paper_values.py",
    REPRO / "generate_supp_rank_table.py",
    REPRO / "generate_uncertainty_supp_table.py",
    REPRO / "generate_pretraining_overlap_table.py",
    REPRO / "generate_cross_benchmark_float.py",
    REPRO / "generate_m_sweep_table.py",
    REPRO / "_cross_benchmark.py",
    REPRO / "build_paper.py",
    REPRO / "check_paper_figures.py",
    ROOT / "scripts" / "studies" / "bootstrap_uncertainty.py",
    ROOT / "scripts" / "studies" / "apd" / "loaders.py",
    REPRO / "figures" / "typed_neighbor_rank_experiment.py",
    REPRO / "figures" / "cross_benchmark_figure.py",
    REPRO / "figures" / "scale_scatter.py",
    REPRO / "generate_distribution_floats.py",
    REPRO / "_distributions.py",
    REPRO / "figures" / "croma_distribution_figure.py",
    REPRO / "generate_pareto_float.py",
    REPRO / "figures" / "croma_pareto_figure.py",
    REPRO / "_rank_pareto.py",
    REPRO / "generate_rank_pareto_float.py",
    REPRO / "figures" / "rank_pareto_figure.py",
]


@pytest.mark.parametrize("script", _PAPER_PIPELINE, ids=lambda p: p.name)
def test_no_paper_script_hardcodes_a_protocol_path(script):
    """Ask ``paper_manifest`` for the run directory; never spell one out.

    A literal ``output/metrics/<protocol>/`` is a second source of truth for which run
    backs the paper, and it goes stale silently -- the generator skips, and the macros it
    feeds vanish without a compile error. See ADR-0010.
    """
    # A concrete protocol is the bug; ``output/metrics/<protocol>/`` in a docstring is not.
    offenders = [
        line.strip()
        for line in script.read_text().splitlines()
        for proto in ("k-star", "median-k")
        if f"output/metrics/{proto}/" in line
    ]
    assert not offenders, (
        f"{script.name} spells out a run directory: {offenders}. "
        f"Use paper_manifest.by_benchmark(...).run_rel / .studies_rel instead."
    )


class TestPandaFloat:
    """The two-panel slide float is generated from both runs at once (ADR-0010)."""

    def test_caption_describes_panels_without_narrating_model_results(self):
        caption = _caption_line(gpt.build_float(ROOT))
        assert r"\textbf{a,}" in caption and r"\textbf{b,}" in caption
        assert "flips" not in caption
        assert r"\code{PRISM}" not in caption

    def test_both_labels_attach_to_the_single_float(self):
        tex = gpt.build_float(ROOT)
        assert tex.count(r"\begin{table}") == 1
        assert r"\label{tab:main-results-panda}" in tex
        assert r"\label{tab:main-results-panda-isup}" in tex

    def test_grading_panel_reports_confounder_accuracy_and_failure_prevalence(self):
        assert [c[0] for c in gpt.PANEL_B] == [c[0] for c in gpt.PANEL_A]

    def test_failure_prevalence_bolds_the_minimum(self):
        tex = gpt.build_float(ROOT)
        prism_row = [ln for ln in tex.splitlines() if ln.startswith("PRISM & 3 & 0.858")][0]
        assert r"\textbf{0.651}" in prism_row
        assert r"\textbf{0.675}" not in tex

    def test_panels_report_a_per_model_operating_point(self):
        """The old hand-authored caption announced a shared k=9 for a k-star protocol."""
        tex = gpt.build_float(ROOT)
        assert "shared operating point" not in tex
        # The caption reports the model-specific k*, not a single shared operating point.
        assert r"model-specific, biologically selected $k^\star$" in tex


#: Floats the prose already cites but nobody has produced yet. Shrink this set; never grow
#: it. Each entry compiles to a "??" in the PDF, so the paper is not submittable until it is
#: empty. Recorded here rather than left to a LaTeX warning nobody reads.
KNOWN_MISSING_LABELS = frozenset({
    "sec:croma-cardinality",
})

_COMMENT = re.compile(r"(?<!\\)%.*")


def _reachable(root: Path) -> list[Path]:
    """Every .tex actually compiled, by walking \\input from main.tex.

    ``\\input`` takes the extension or leaves it off, and both forms compile. Appending
    ".tex" unconditionally made every file reached by the explicit form invisible here --
    which hid all of ``results_figures.tex``, the whole main-text figure set, from the
    dangling-reference and duplicate-label checks below.
    """
    seen: list[Path] = []

    def walk(rel: str) -> None:
        path = root / "paper" / (rel if rel.endswith(".tex") else f"{rel}.tex")
        if not path.exists() or path in seen:
            return
        seen.append(path)
        body = _COMMENT.sub("", path.read_text())
        for nested in re.findall(r"\\input\{([^}]+)\}", body):
            walk(nested)

    walk("main")
    return seen


def _labels_and_refs(files: list[Path]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    labels: dict[str, list[str]] = {}
    refs: dict[str, list[str]] = {}
    for path in files:
        body = _COMMENT.sub("", path.read_text())
        for lab in re.findall(r"\\label\{([^}]+)\}", body):
            labels.setdefault(lab, []).append(path.name)
        for ref in re.findall(r"\\(?:auto|c|C)?ref\{([^}]+)\}", body):
            refs.setdefault(ref, []).append(path.name)
    return labels, refs


class TestCrossReferences:
    """A \\ref to a label no compiled file defines silently becomes "??" in the PDF.

    This caught a live near-miss: enabling in-table CIs makes the results caption cite
    ``tab:bootstrap-uncertainty``, whose supplement is commented out of ``supp.tex``.
    """

    def _live(self):
        files = _reachable(ROOT)
        if len(files) < 2:
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        return _labels_and_refs(files)

    def test_no_new_dangling_references(self):
        labels, refs = self._live()
        dangling = {r for r in refs if r not in labels} - KNOWN_MISSING_LABELS
        assert not dangling, f"new dangling \\ref(s): {sorted(dangling)}"

    def test_known_missing_labels_are_still_missing(self):
        """Delete an entry from KNOWN_MISSING_LABELS once its float lands."""
        labels, _ = self._live()
        landed = KNOWN_MISSING_LABELS & set(labels)
        assert not landed, f"these floats now exist; drop them from KNOWN_MISSING_LABELS: {sorted(landed)}"

    def test_no_duplicate_labels(self):
        labels, _ = self._live()
        dupes = {k: v for k, v in labels.items() if len(v) > 1}
        assert not dupes, f"duplicate \\label(s): {dupes}"


def test_every_live_caption_is_within_the_350_word_limit():
    """Journal guidance caps each figure or table legend at 350 words."""
    captions = []
    for path in _reachable(ROOT):
        captions.extend(
            (path.name, line.strip())
            for line in path.read_text().splitlines()
            if r"\caption{" in line
        )
    if not captions:
        pytest.skip("paper/ is git-ignored and absent in this checkout")

    over = {}
    for filename, caption in captions:
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", caption)
        if len(words) > 350:
            over[filename] = len(words)
    assert not over, f"caption(s) exceed 350 words: {over}"


def _provenance_frame(overrides: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """The Section 3.4 story in miniature: leader, its probe-space twin, and two foils.

    Matches the live TCGA-4x4 ordering -- ``Midnight-12k`` best on biology and mid-pack on
    centre decodability, ``CONCH`` least centre-decodable but weaker on biology, ``CONCHv1.5``
    the nearest neighbour once the two accuracies are differenced.
    """
    rows = {
        "Midnight-12k": {"bio": 0.882, "conf": 0.559, "croma": 0.396},
        "CONCHv1.5": {"bio": 0.811, "conf": 0.492, "croma": 0.153},
        "CONCH": {"bio": 0.790, "conf": 0.487, "croma": 0.146},
        "H-optimus-1": {"bio": 0.878, "conf": 0.676, "croma": 0.088},
        "GenBio-PathFM": {"bio": 0.851, "conf": 0.695, "croma": 0.155},
        "UNI2-h": {"bio": 0.847, "conf": 0.737, "croma": 0.075},
        "H-optimus-0": {"bio": 0.846, "conf": 0.692, "croma": 0.054},
        CONTROL_MODEL: {"bio": 0.607, "conf": 0.579, "croma": 0.006},
    }
    for model, changes in (overrides or {}).items():
        rows[model].update(changes)
    return pd.DataFrame([
        {"model": m, "bio_knn_bacc": v["bio"], "confounder_knn_bacc": v["conf"], "croma": v["croma"]}
        for m, v in rows.items()
    ])


class TestProvenanceClaims:
    """Section 3.4's mechanism argument, which the data once flatly contradicted.

    The prose asserted that ``Midnight-12k`` had "the lowest biological k-NN accuracy among
    the leading encoders" and "the least decodable [centre] of all 16 models". After the
    TCGA run was corrected from eight medical centres to PathoROB's four in-domain ones, it
    had the *highest* biological accuracy and the third-least-decodable centre -- inverting
    the paragraph's conclusion. Nothing caught it, because a paragraph asserts and a run
    merely changes. Each clause is now a predicate; each predicate is fed its counterexample.
    """

    def test_the_live_panel_supports_every_clause(self):
        assert len(gpv._provenance_macros(_provenance_frame())) == 16

    def test_losing_the_croma_lead_raises(self):
        frame = _provenance_frame({"GenBio-PathFM": {"croma": 0.5}})
        with pytest.raises(CaptionClaimError, match="CRoMa leader"):
            gpv._provenance_macros(frame)

    def test_losing_the_biology_crown_raises(self):
        """"not bought by surrendering class separation" needs the crown, not a good rank."""
        frame = _provenance_frame({"H-optimus-1": {"bio": 0.95}})
        with pytest.raises(CaptionClaimError, match="highest biological"):
            gpv._provenance_macros(frame)

    def test_becoming_the_least_centre_decodable_model_raises(self):
        """The paragraph's second half argues the lead is *not* pure centre-invariance."""
        frame = _provenance_frame({"Midnight-12k": {"conf": 0.30}})
        with pytest.raises(CaptionClaimError, match="not bought by centre-invariance alone"):
            gpv._provenance_macros(frame)

    def test_a_top_biology_peer_hiding_its_centre_better_raises(self):
        frame = _provenance_frame({"Midnight-12k": {"conf": 0.70}})
        with pytest.raises(CaptionClaimError, match="alone resists centre decoding"):
            gpv._provenance_macros(frame)

    def test_the_twin_drifting_to_another_model_raises(self):
        """CONCH's gap moved onto Midnight's; the named twin is no longer the nearest."""
        frame = _provenance_frame({"CONCH": {"conf": 0.790 - 0.3227}})
        with pytest.raises(CaptionClaimError, match="nearest neighbour"):
            gpv._provenance_macros(frame)

    def test_the_twin_catching_up_raises(self):
        """The whole point is a wide CRoMa gap at equal probe coordinates."""
        frame = _provenance_frame({"CONCHv1.5": {"croma": 0.30}})
        with pytest.raises(CaptionClaimError, match="under 2.0x"):
            gpv._provenance_macros(frame)

    def test_a_missing_model_raises_rather_than_reindexing(self):
        frame = _provenance_frame()
        frame = frame[frame["model"] != "CONCHv1.5"]
        with pytest.raises(CaptionClaimError, match="absent from the ranked panel"):
            gpv._provenance_macros(frame)

    def test_the_control_never_enters_the_claims(self):
        """Give the control the best biology and the least decodable centre; nothing moves.

        DINOv2-B is a floor, not a competitor: it is excluded from every cross-model claim,
        so promoting it must not steal the crown from the model the paragraph names.
        """
        frame = _provenance_frame({CONTROL_MODEL: {"bio": 0.99, "conf": 0.10}})
        assert len(gpv._provenance_macros(frame)) == 16


class TestProvenanceOverlapClaims:
    """The Tolkach half: the lead must survive dropping the benchmark's TCGA cohort."""

    def _rows(self, overrides: dict[str, dict[str, float]] | None = None):
        base = {
            "Midnight-12k": {"rest": 0.5986, "tcga": 0.8172, "boost": 2.496},
            "CONCH": {"rest": 0.4391, "tcga": 0.5215, "boost": 1.240},
            "H0-mini": {"rest": 0.3780, "tcga": 0.5301, "boost": 1.470},
        }
        for model, changes in (overrides or {}).items():
            base[model].update(changes)
        return pd.DataFrame([{"model": m, **v} for m, v in base.items()])

    def _patched(self, monkeypatch, frame):
        import _overlap

        monkeypatch.setattr(_overlap, "rows", lambda include_control=True: frame)
        return gpv._provenance_overlap_macros()

    def test_the_live_numbers_support_both_clauses(self, monkeypatch):
        assert len(self._patched(monkeypatch, self._rows())) == 7

    def test_losing_the_largest_boost_raises(self, monkeypatch):
        with pytest.raises(CaptionClaimError, match="largest TCGA boost"):
            self._patched(monkeypatch, self._rows({"H0-mini": {"boost": 3.0}}))

    def test_losing_the_tcga_free_lead_raises(self, monkeypatch):
        """This is the "amplified, rather than created" sentence. Without it, "created"."""
        with pytest.raises(CaptionClaimError, match="survives dropping the TCGA cohort"):
            self._patched(monkeypatch, self._rows({"CONCH": {"rest": 0.7}}))

    def test_the_two_runner_ups_are_not_conflated(self, monkeypatch):
        """Runner-up by boost (H0-mini) is a different model from runner-up by subset CRoMa
        (CONCH). The prose names both, in different sentences."""
        lines = "\n".join(self._patched(monkeypatch, self._rows()))
        assert r"\newcommand{\ProvenanceTolkachBoostRunnerUpModel}{H0-mini}" in lines
        assert r"\newcommand{\ProvenanceTolkachRestRunnerUpModel}{CONCH}" in lines


class TestApdControlExclusion:
    """Both APD captions assert the natural-image control is excluded. It is a claim.

    ``loaders.ranked`` enforces it today, but the sentence would survive that helper's
    removal: the rho values would shift, ``n_models`` would tick up by one, and nothing
    would say so. The control is doubly flattered by APD -- high CRoMa because its biology
    is poor, and a lenient *relative* drop because it has little accuracy to lose -- so its
    presence would inflate every rho in the table.
    """

    def _apd(self, tmp_path, joined_models, n_by_scope):
        import _apd as m

        joined = pd.DataFrame(
            [{"dataset": ds, "model": mo} for ds, models in joined_models.items()
             for mo in models]
        )
        path = tmp_path / "apd_metrics_joined.csv"
        joined.to_csv(path, index=False)
        corr = pd.DataFrame(
            [{"target": "apd_id", "metric": "croma", "scope": s, "n": n, "spearman": 0.9}
             for s, n in n_by_scope.items()]
        )
        return m, path, m.Apd(corr=corr)

    def _rosters(self, *, with_control=True):
        pathology = [f"M{i}" for i in range(20)]
        models = [*pathology, CONTROL_MODEL] if with_control else pathology
        return {ds: models for ds in ["camelyon", "tcga_4x4", "tolkach"]}

    def test_the_live_artifacts_satisfy_the_claim(self):
        """Guards the real study output, not a fixture: this is what the captions ship."""
        import _apd as m

        if not m.CORRELATION_CSV.exists() or not m.JOINED_CSV.exists():
            pytest.skip("APD study has not been run in this checkout")
        m.load().assert_control_excluded()

    def test_ranking_over_the_full_panel_raises(self, tmp_path, monkeypatch):
        """The exact regression: someone drops ``ranked()`` from ``corr_block``."""
        m, path, apd = self._apd(tmp_path, self._rosters(),
                                 {"camelyon": 21, "tcga_4x4": 20, "tolkach": 20})
        monkeypatch.setattr(m, "JOINED_CSV", path)
        with pytest.raises(CaptionClaimError, match="21 of 21 joined models"):
            apd.assert_control_excluded()

    def test_a_control_absent_from_the_join_raises(self, tmp_path, monkeypatch):
        """The join keeps the control on record. If it stops, the sentence is vacuous and
        the count coincidentally still checks out -- so count alone is not enough."""
        m, path, apd = self._apd(tmp_path, self._rosters(with_control=False),
                                 {"camelyon": 19, "tcga_4x4": 19, "tolkach": 19})
        monkeypatch.setattr(m, "JOINED_CSV", path)
        with pytest.raises(CaptionClaimError, match="missing from the camelyon join"):
            apd.assert_control_excluded()

    def test_the_ranked_panel_passes(self, tmp_path, monkeypatch):
        m, path, apd = self._apd(tmp_path, self._rosters(),
                                 {"camelyon": 20, "tcga_4x4": 20, "tolkach": 20})
        monkeypatch.setattr(m, "JOINED_CSV", path)
        apd.assert_control_excluded()


#: Uppercase-leading control sequences LaTeX and its packages supply. The paper's own macros
#: are all uppercase-leading CamelCase (\CamelyonCromaSpan, \ProvenanceBioBacc), so scanning
#: for that shape finds them without wading through \textbf, \ref, \begin and friends.
_LATEX_UPPERCASE = frozenset({"Delta", "Rightarrow", "S", "P", "LaTeX", "TeX", "FloatBarrier"})


def _macro_defs_and_uses(files: list[Path]) -> tuple[set[str], dict[str, list[str]]]:
    defs: set[str] = set()
    uses: dict[str, list[str]] = {}
    for path in files:
        body = _COMMENT.sub("", path.read_text())
        defs |= set(re.findall(r"\\(?:re)?newcommand\*?\{?\\([A-Za-z]+)\}?", body))
        for name in re.findall(r"\\([A-Z][A-Za-z]*)", body):
            uses.setdefault(name, []).append(path.name)
    return defs, uses


class TestMacros:
    r"""Prose cites macros that a *skipped* generator block would never have written.

    ``generate_paper_values.py`` warns-and-skips whole macro families when their study
    artifact is absent (APD, bootstrap, SS-shell, pretraining overlap). The prose keeps
    citing them, and an undefined ``\ProvenanceTolkachRest`` is not a wrong number -- it is a
    LaTeX error, in a document nothing in CI compiles. Nobody would see it until a build.
    """

    def _live(self):
        files = _reachable(ROOT)
        if len(files) < 2:
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        return _macro_defs_and_uses(files)

    def test_every_macro_the_paper_cites_is_defined(self):
        defs, uses = self._live()
        undefined = {n: sorted(set(f)) for n, f in uses.items()
                     if n not in defs and n not in _LATEX_UPPERCASE}
        assert not undefined, f"undefined macro(s) cited in the paper: {undefined}"

    def test_no_macro_is_defined_twice(self):
        r"""A second ``\newcommand`` for a live macro is a LaTeX error, not a redefinition."""
        files = _reachable(ROOT)
        if len(files) < 2:
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        seen: dict[str, list[str]] = {}
        for path in files:
            body = _COMMENT.sub("", path.read_text())
            for name in re.findall(r"\\newcommand\*?\{?\\([A-Za-z]+)\}?", body):
                seen.setdefault(name, []).append(path.name)
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        assert not dupes, f"macro(s) defined twice: {dupes}"

    def test_the_provenance_family_is_fully_cited(self):
        """Every ``\\Provenance*`` macro exists because Section 3.4 names it.

        An unused one means the paragraph was edited away from a claim while its guard --
        the reason the generator raises when the ordering inverts -- stayed behind, still
        passing, protecting nothing.
        """
        defs, uses = self._live()
        emitted = {n for n in defs if n.startswith("Provenance")}
        if not emitted:
            pytest.skip("generated_values.tex has no provenance block")
        assert emitted <= set(uses), f"emitted but never cited: {sorted(emitted - set(uses))}"


class TestBootstrapCoupling:
    """Rendering CIs is a manifest decision, not an accident of what is on disk."""

    def test_cis_are_off_unless_the_manifest_asks(self, tmp_path):
        metrics = _write_run(tmp_path, HEALTHY)
        # A bootstrap artifact sitting beside metrics.csv must not change the table.
        pd.DataFrame({
            "model": [r["model"] for r in HEALTHY],
            "croma_lo": [-1.0] * len(HEALTHY),
            "croma_hi": [1.0] * len(HEALTHY),
        }).to_csv(metrics.parent / "bootstrap_uncertainty.csv", index=False)
        tex = grt.build_table(_entry(), metrics)
        assert "scriptsize" not in tex
        assert "confidence interval" not in tex

    def test_asking_for_cis_without_the_artifact_is_an_error(self, tmp_path):
        metrics = _write_run(tmp_path, HEALTHY)
        with pytest.raises(FileNotFoundError, match="bootstrap_uncertainty.csv"):
            grt.build_table(_entry(with_ci=True), metrics)


@pytest.mark.parametrize("entry", pm.rendered(), ids=lambda e: e.prefix)
def test_paper_table_is_not_stale(entry):
    """The .tex on disk equals what the current metrics.csv renders. See ADR-0010."""
    out = ROOT / entry.out_tex
    metrics = ROOT / entry.metrics_rel
    if not out.parent.exists():
        pytest.skip("paper/ is git-ignored and absent in this checkout")
    if not metrics.exists():
        pytest.skip(f"{entry.metrics_rel} absent; run scripts/repro/run_benchmarks.sh")
    assert out.exists(), f"{entry.out_tex} missing; run scripts/repro/build_paper.py"
    assert out.read_text() == grt.render(entry, ROOT), (
        f"{entry.out_tex} is stale relative to {entry.metrics_rel}. "
        f"Run scripts/repro/build_paper.py; do not hand-edit."
    )


def test_panda_float_is_not_stale():
    if not gpt.OUT.parent.exists():
        pytest.skip("paper/ is git-ignored and absent in this checkout")
    assert gpt.OUT.read_text() == gpt.build(ROOT), (
        "paper/sections/supp/table_panda_isup.tex is stale. Run scripts/repro/build_paper.py; "
        "do not hand-edit (its prose lives in paper/sections/supp/panda.tex, which \\inputs it)."
    )


def test_m_sweep_table_is_not_stale():
    """tab:m-sweep re-derives its rho and confounder-dominant counts from the live sweeps."""
    if not gms.OUT.parent.exists():
        pytest.skip("paper/ is git-ignored and absent in this checkout")
    sweep = ROOT / gms.by_benchmark("pathorob-camelyon").run_rel / "results" / "croma_m_sweep_metrics.csv"
    if not sweep.exists():
        pytest.skip("CRoMa m-sweep metrics absent; re-run the benchmark with the sweep enabled")
    assert gms.OUT.read_text() == gms.build(), (
        "paper/sections/supp/m_sweep.tex is stale relative to the croma_m_sweep_metrics.csv "
        "runs. Run scripts/repro/build_paper.py; do not hand-edit."
    )


class TestCrossBenchmarkCaption:
    """The legend defines the plotted encodings without turning trajectories into prose."""

    def test_the_control_holds_no_rank(self):
        cb = _cb.load()
        assert CONTROL_MODEL not in cb.croma.index
        assert cb.ranks.index.equals(cb.croma.index)

    def test_the_dagger_set_comes_from_metadata(self):
        """A hard-coded set of five outlived the metadata's nine."""
        md = pd.read_csv(_cb.METADATA)
        expected = set(md.loc[md["tcga_exposed"], "model"])
        cb = _cb.load()
        assert cb.exposed == frozenset(expected & set(cb.croma.index))
        assert cb.exposed, "no exposed model: the dagger legend would mark nothing"

    def test_ranks_are_dense_and_start_at_one(self):
        cb = _cb.load()
        for label in cb.labels:
            assert sorted(cb.ranks[label]) == list(range(1, cb.n_models + 1))

    def test_caption_defines_encodings_without_named_trajectory_claims(self):
        caption = _caption_line(gcb.build())
        assert "filled" in caption and "hollow" in caption
        assert "Dashed trajectories" in caption and r"$\dagger$" in caption
        assert r"\code{Midnight-12k}" not in caption
        assert r"\code{CONCH}" not in caption

    def test_float_is_not_stale(self):
        if not gcb.OUT.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        assert gcb.OUT.read_text() == gcb.build(), (
            "paper/sections/supp_cross_benchmark.tex is stale. Run scripts/repro/build_paper.py."
        )


def _dist_frames(overrides: dict[str, dict[str, float]] | None = None):
    """The tail analysis in miniature: a seven-encoder stand-in for the Camelyon run.

    Mirrors the live Camelyon ordering so the roster ranks best-median-first exactly as the
    figure draws it, and contains every encoder the caption names (the same-median pair, the
    hidden-tail example, the control, and \\code{GPFM} -- the one CAMELYON-exposed encoder the
    Pareto caption daggers). ``croma``/``ltm`` feed the metrics frame; ``f0`` is materialised as
    a per-sample column (``build`` recomputes F(0) from it, as from the real run).
    """
    base = {
        "Virchow2": {"croma": 0.199, "ltm": -0.106, "f0": 0.129},
        "CONCH": {"croma": 0.196, "ltm": -0.202, "f0": 0.225},
        "Midnight-12k": {"croma": 0.108, "ltm": -0.349, "f0": 0.354},
        CONTROL_MODEL: {"croma": 0.050, "ltm": -0.184, "f0": 0.345},
        "H-optimus-0": {"croma": 0.045, "ltm": -0.154, "f0": 0.315},
        "GPFM": {"croma": -0.102, "ltm": -0.363, "f0": 0.627},
        "Hibou-L": {"croma": -0.443, "ltm": -0.659, "f0": 0.993},
    }
    for model, changes in (overrides or {}).items():
        base[model].update(changes)
    metrics = pd.DataFrame(
        [{"model": m, "croma": v["croma"], "croma_ltm_alpha": v["ltm"]} for m, v in base.items()]
    )
    n = 1000
    col = f"croma_m{HEADLINE_M}"
    per_sample = pd.DataFrame(
        [
            {"model": m, col: (-0.5 if i < round(v["f0"] * n) else 0.5)}
            for m, v in base.items()
            for i in range(n)
        ]
    )
    return metrics, per_sample


def _panda_frames(overrides: dict[str, dict[str, float]] | None = None):
    """The slide-level tail analysis in miniature: the four PANDA whole-slide encoders, no control.

    Mirrors the live PANDA ordering -- PRISM biology-dominant at the median, MOOZY at the
    boundary, then TITAN and Prov-GigaPath -- and the named contrast the caption makes: PRISM and
    MOOZY differ sharply in median and F(0) yet share a worst-decile severity. Same frame shape
    as ``_dist_frames`` (``croma``/``ltm`` feed the metrics frame; ``f0`` a per-sample column).
    """
    base = {
        "PRISM": {"croma": 0.257, "ltm": -0.390, "f0": 0.288},
        "MOOZY": {"croma": -0.016, "ltm": -0.412, "f0": 0.535},
        "TITAN": {"croma": -0.295, "ltm": -0.602, "f0": 0.895},
        "Prov-GigaPath": {"croma": -0.413, "ltm": -0.592, "f0": 0.990},
    }
    for model, changes in (overrides or {}).items():
        base[model].update(changes)
    metrics = pd.DataFrame(
        [{"model": m, "croma": v["croma"], "croma_ltm_alpha": v["ltm"]} for m, v in base.items()]
    )
    n = 1000
    col = f"croma_m{HEADLINE_M}"
    per_sample = pd.DataFrame(
        [
            {"model": m, col: (-0.5 if i < round(v["f0"] * n) else 0.5)}
            for m, v in base.items()
            for i in range(n)
        ]
    )
    return metrics, per_sample


class TestDistributionFloats:
    """Ridgeline legends define ordering and reference encodings, not observed tails."""

    def test_the_roster_ranks_best_median_first(self):
        dist = _dist.build(*_dist_frames())
        assert [m.name for m in dist.models] == [
            "Virchow2", "CONCH", "Midnight-12k", CONTROL_MODEL, "H-optimus-0", "GPFM", "Hibou-L"
        ]

    def test_the_live_caption_builds(self):
        try:
            dist = _dist.load()
        except FileNotFoundError:
            pytest.skip("Camelyon run absent; run scripts/repro/run_benchmarks.sh")
        assert r"\label{fig:croma-distribution}" in gdf.build_figure(dist)

    def test_main_caption_is_invariant_to_distribution_results(self):
        original = gdf.build_figure(_dist.build(*_dist_frames()))
        altered = gdf.build_figure(_dist.build(*_dist_frames({
            "Virchow2": {"croma": -0.7, "ltm": 0.2},
            "CONCH": {"croma": 0.8, "ltm": 0.3},
            "Midnight-12k": {"croma": -0.6, "ltm": 0.4},
        })))
        assert _caption_line(original) == _caption_line(altered)
        assert r"\code{Virchow2}" not in _caption_line(original)

    def test_the_control_flag_backs_the_caption(self):
        dist = _dist.build(*_dist_frames())
        assert _dist.assert_control_is_the_control(dist).name == CONTROL_MODEL

    def test_f0_is_read_on_the_bounded_margin_not_the_raw_ratio(self):
        """A run stored as the raw distance ratio (never negative) must not read as 0% fragile.

        Re-express the per-sample column as the ratio r = (1+margin)/(1-margin): every value is
        positive, so a naive ``< 0`` count would put F(0)=0 everywhere. ``build`` normalises
        first, so F(0) is unchanged and the tail predicate still fires on the flipped tail.
        """
        metrics, per_sample = _dist_frames()
        col = f"croma_m{HEADLINE_M}"
        per_sample[col] = (1.0 + per_sample[col]) / (1.0 - per_sample[col])  # margin -> ratio
        dist = _dist.build(metrics, per_sample)
        assert dist.by_name["Hibou-L"].f0 > 0.9  # not silently zeroed by the ratio storage

    def test_the_float_is_not_stale(self):
        fig = ROOT / "paper" / "sections" / "results_figure_distribution.tex"
        if not fig.parent.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        if not (ROOT / _dist.CAMELYON.metrics_rel).exists():
            pytest.skip("Camelyon run absent; run scripts/repro/run_benchmarks.sh")
        dist = _dist.load(ROOT)
        assert fig.read_text() == gdf.build_figure(dist), (
            "paper/sections/results_figure_distribution.tex is stale. "
            "Run scripts/repro/build_paper.py."
        )

    def test_the_live_supp_captions_build(self):
        """Both supplementary floats build from their live runs, each with its own label."""
        try:
            blocks = gdf.build_supp(ROOT)
        except FileNotFoundError:
            pytest.skip("TCGA-4x4/Tolkach runs absent; run scripts/repro/run_benchmarks.sh")
        assert r"\label{fig:croma-distribution-tcga4x4}" in blocks
        assert r"\label{fig:croma-distribution-tolkach}" in blocks

    def test_supp_caption_does_not_depend_on_tail_sign(self):
        original = _dist.build(*_dist_frames())
        altered = _dist.build(*_dist_frames({"Hibou-L": {"ltm": 0.05}}))
        entry = _dist.SUPP_BENCHMARKS[0]
        assert _caption_line(gdf.build_supp_figure(entry, original)) == _caption_line(
            gdf.build_supp_figure(entry, altered)
        )

    def test_the_supp_float_is_not_stale(self):
        fig = ROOT / "paper" / "sections" / "supp" / "figure_distributions.tex"
        if not fig.parent.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        if not (ROOT / _dist.SUPP_BENCHMARKS[0].metrics_rel).exists():
            pytest.skip("supp tail runs absent; run scripts/repro/run_benchmarks.sh")
        assert fig.read_text() == gdf.build_supp(ROOT), (
            "paper/sections/supp/figure_distributions.tex is stale. "
            "Run scripts/repro/build_paper.py."
        )


class TestSlideDistributionFloat:
    """The slide-level legend remains descriptive and the panel excludes the tile control."""

    def test_the_slide_roster_ranks_best_median_first(self):
        dist = _dist.build(*_panda_frames())
        assert [m.name for m in dist.models] == ["PRISM", "MOOZY", "TITAN", "Prov-GigaPath"]

    def test_a_natural_image_control_in_the_slide_panel_raises(self):
        """DINOv2-B produces no slide embedding; if one enters the run the caption's count
        (\\PandaRankedNModels{} whole-slide encoders, no control band) is wrong."""
        metrics, per_sample = _panda_frames()
        col = f"croma_m{HEADLINE_M}"
        metrics = pd.concat(
            [metrics, pd.DataFrame([{"model": CONTROL_MODEL, "croma": 0.05, "croma_ltm_alpha": -0.2}])],
            ignore_index=True,
        )
        per_sample = pd.concat(
            [per_sample, pd.DataFrame([{"model": CONTROL_MODEL, col: 0.5}])], ignore_index=True
        )
        with pytest.raises(_dist.CaptionClaimError, match="only whole-slide encoders"):
            _dist.assert_no_natural_image_control(_dist.build(metrics, per_sample))

    def test_slide_caption_does_not_depend_on_tail_results(self):
        original = _dist.build(*_panda_frames())
        altered = _dist.build(*_panda_frames({"Prov-GigaPath": {"ltm": 0.05}}))
        assert _caption_line(gdf.build_panda_figure(original)) == _caption_line(
            gdf.build_panda_figure(altered)
        )

    def test_the_live_caption_builds(self):
        try:
            dist = _dist.load(ROOT, _dist.PANDA)
        except FileNotFoundError:
            pytest.skip("PANDA run absent; run scripts/repro/run_benchmarks.sh")
        assert r"\label{fig:croma-distribution-panda}" in gdf.build_panda_figure(dist)

    def test_the_slide_float_is_not_stale(self):
        fig = ROOT / "paper" / "sections" / "supp" / "figure_distribution_panda.tex"
        if not fig.parent.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        if not (ROOT / _dist.PANDA.metrics_rel).exists():
            pytest.skip("PANDA run absent; run scripts/repro/run_benchmarks.sh")
        dist = _dist.load(ROOT, _dist.PANDA)
        assert fig.read_text() == gdf.build_panda_figure(dist), (
            "paper/sections/supp/figure_distribution_panda.tex is stale. "
            "Run scripts/repro/build_paper.py."
        )


class TestParetoFloat:
    """Pareto legends define axes and encodings without naming observed leaders."""

    def test_the_caption_builds_and_carries_the_label(self):
        tex = gpf.build_figure(_dist.build(*_dist_frames()))
        assert r"\label{fig:croma-pareto}" in tex
        caption = _caption_line(tex)
        assert "Ringed points" in caption and "shaded points" in caption
        assert r"\code{CONCH}" not in caption
        assert r"\code{Virchow2}" not in caption

    def test_caption_is_invariant_to_frontier_membership(self):
        original = gpf.build_figure(_dist.build(*_dist_frames()))
        altered = gpf.build_figure(
            _dist.build(*_dist_frames({"Midnight-12k": {"croma": 0.30, "ltm": -0.05}}))
        )
        assert _caption_line(original) == _caption_line(altered)

    def test_the_float_is_not_stale(self):
        fig = ROOT / "paper" / "sections" / "results_figure_pareto.tex"
        if not fig.parent.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        if not (ROOT / _dist.CAMELYON.metrics_rel).exists():
            pytest.skip("Camelyon run absent; run scripts/repro/run_benchmarks.sh")
        assert fig.read_text() == gpf.build_figure(_dist.load(ROOT)), (
            "paper/sections/results_figure_pareto.tex is stale. "
            "Run scripts/repro/build_paper.py."
        )

    # --- the two supplementary floats and the transferable two-axis guard -------------------

    def test_the_supp_caption_builds_without_naming_leaders(self):
        dist = _dist.build(*_dist_frames({"H-optimus-0": {"ltm": -0.05}}))
        tex = gpf.build_supp_figure(_dist.SUPP_BENCHMARKS[0], dist)
        assert r"\label{fig:croma-pareto-tcga4x4}" in tex
        caption = _caption_line(tex)
        assert r"\code{Virchow2}" not in caption
        assert r"\code{H-optimus-0}" not in caption
        assert "does not also hold the mildest" not in caption

    def test_the_live_supp_captions_build(self):
        """Both supplementary floats build from their live runs, each with its own label."""
        try:
            blocks = gpf.build_supp(ROOT)
        except FileNotFoundError:
            pytest.skip("TCGA-4x4/Tolkach runs absent; run scripts/repro/run_benchmarks.sh")
        assert r"\label{fig:croma-pareto-tcga4x4}" in blocks
        assert r"\label{fig:croma-pareto-tolkach}" in blocks

    def test_the_supp_float_is_not_stale(self):
        fig = ROOT / "paper" / "sections" / "supp" / "figure_pareto.tex"
        if not fig.parent.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        if not (ROOT / _dist.SUPP_BENCHMARKS[0].metrics_rel).exists():
            pytest.skip("supp tail runs absent; run scripts/repro/run_benchmarks.sh")
        assert fig.read_text() == gpf.build_supp(ROOT), (
            "paper/sections/supp/figure_pareto.tex is stale. "
            "Run scripts/repro/build_paper.py."
        )


def _rank_pareto(rows: dict[str, dict[str, list[float]]], *, exposed=(), adversarial="Camelyon"):
    """Build a RankPareto from per-model margins/LTMs across three benchmarks.

    ``rows`` maps model -> {"med": [m0, m1, m2], "ltm": [l0, l1, l2]}. Ranks are derived exactly
    as ``_rank_pareto.load()`` does (median: higher = better; tail: higher, less-negative LTM =
    milder = better), so the frontier geometry the predicates read matches the live loader. The
    labels mirror the live short names, so ``adversarial`` defaults to a column that exists.
    """
    labels = ["Camelyon", "TCGA", "Tolkach"]
    med = pd.DataFrame({lab: {m: rows[m]["med"][i] for m in rows} for i, lab in enumerate(labels)})
    ltm = pd.DataFrame({lab: {m: rows[m]["ltm"][i] for m in rows} for i, lab in enumerate(labels)})
    return _rp.RankPareto(
        medians=med,
        median_ranks=med.rank(ascending=False, method="first").astype(int),
        tail_ranks=ltm.rank(ascending=False, method="first").astype(int),
        exposed=frozenset(exposed),
        adversarial=adversarial,
    )


#: A compact rank-aggregate panel with several frontier and exposure-marker states.
_RANK_HEALTHY = {
    "CONCH":         {"med": [0.30, 0.30, 0.28], "ltm": [-0.11, -0.10, -0.09]},
    "GenBio-PathFM": {"med": [0.25, 0.16, 0.15], "ltm": [-0.08, -0.08, -0.07]},
    "H-optimus-1":   {"med": [0.20, 0.10, 0.12], "ltm": [-0.05, -0.05, -0.05]},
    "Midnight-12k":  {"med": [0.11, 0.40, 0.58], "ltm": [-0.35, -0.25, -0.29]},
    "Phikon":        {"med": [0.15, 0.02, 0.05], "ltm": [-0.30, -0.20, -0.18]},
}
_RANK_EXPOSED = ("Midnight-12k", "GenBio-PathFM")


class TestRankParetoFloat:
    """The aggregate legend defines rank axes and encodings without narrating outcomes."""

    def test_the_healthy_panel_has_a_frontier_and_exposure_markers(self):
        rp = _rank_pareto(_RANK_HEALTHY, exposed=_RANK_EXPOSED)
        assert _rp.assert_exposure_marked(rp) == frozenset(_RANK_EXPOSED)
        assert set(rp.frontier) == {"CONCH", "GenBio-PathFM", "H-optimus-1"}

    def test_no_exposed_model_raises(self):
        with pytest.raises(_rp.CaptionClaimError, match="TCGA-exposed"):
            _rp.assert_exposure_marked(_rank_pareto(_RANK_HEALTHY, exposed=()))

    def test_the_live_caption_builds(self):
        try:
            tex = grp.build()
        except FileNotFoundError:
            pytest.skip("tile runs absent; run scripts/repro/run_benchmarks.sh")
        assert r"\label{fig:croma-pareto-rank}" in tex
        caption = _caption_line(tex)
        assert "mean rank across benchmarks" in caption
        assert r"\code{Midnight-12k}" not in caption
        assert "in-distribution artefact" not in caption

    def test_the_float_is_not_stale(self):
        fig = ROOT / "paper" / "sections" / "supp" / "figure_rank_pareto.tex"
        if not fig.parent.exists():
            pytest.skip("paper/ is git-ignored and absent in this checkout")
        if not (ROOT / pm.by_benchmark("pathorob-camelyon").metrics_rel).exists():
            pytest.skip("tile runs absent; run scripts/repro/run_benchmarks.sh")
        assert fig.read_text() == grp.build(), (
            "paper/sections/supp/figure_rank_pareto.tex is stale. "
            "Run scripts/repro/build_paper.py."
        )
