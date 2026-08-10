"""Publication contracts for the expanded PathoROB result set (issue #133)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
REPRO = ROOT / "scripts/repro"
BENCH = ROOT / "scripts/bench"
for path in (REPRO, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_paper_values as paper_values  # noqa: E402
import generate_pretraining_overlap_table as overlap_table  # noqa: E402
import generate_supp_rank_table as typed_table  # noqa: E402
from _rank_pareto import RankPareto  # noqa: E402

EXPECTED_PUBLIC_COHORTS = {"camelyon", "tcga-4x4", "tolkach-esca"}
EXPECTED_PANEL = {
    "CONCH",
    "CONCHv1.5",
    "DINOv2-B",
    "GPFM",
    "GenBio-PathFM",
    "H-optimus-0",
    "H-optimus-1",
    "H0-mini",
    "Hibou-B",
    "Hibou-L",
    "MUSK",
    "Mascaret",
    "Midnight-12k",
    "Phaet",
    "Phikon",
    "Phikon-v2",
    "Prost40M",
    "Prov-GigaPath",
    "RudolfV 2",
    "RudolfV 2-B",
    "RudolfV 2-S",
    "UNI",
    "UNI2-h",
    "Virchow",
    "Virchow2",
    "mSTAR",
}
EXPECTED_HISTORICAL_TYPED_PANEL = EXPECTED_PANEL - {
    "Mascaret",
    "Phaet",
    "RudolfV 2",
    "RudolfV 2-B",
    "RudolfV 2-S",
}


def test_public_export_is_exactly_three_cohorts_and_25_plus_control() -> None:
    provenance = json.loads((ROOT / "results/PROVENANCE.json").read_text())
    aggregate = pd.read_csv(ROOT / "results/cross_benchmark.csv")

    assert set(provenance["cohorts"]) == EXPECTED_PUBLIC_COHORTS
    assert provenance["roster"] == 26
    assert set(aggregate["model"]) == EXPECTED_PANEL
    assert aggregate.loc[aggregate["is_control"], "model"].tolist() == ["DINOv2-B"]
    assert len(aggregate.loc[~aggregate["is_control"]]) == 25
    assert (
        aggregate.loc[aggregate["is_control"], ["mean_rank", "croma_rank", "ltm_rank"]]
        .isna()
        .all()
        .all()
    )
    assert not aggregate.loc[aggregate["is_control"], "on_frontier"].any()
    assert aggregate.loc[~aggregate["is_control"], "croma_rank"].max() <= 25
    assert aggregate.loc[~aggregate["is_control"], "ltm_rank"].max() <= 25
    assert "tcga-2x2" not in provenance["files"]


def test_public_results_section_carries_only_the_three_public_cohorts() -> None:
    """One page per public cohort, each carrying its table and Pareto panel.

    The invariant is the cohort set: exactly the three public cohorts in the results
    toctree, and no supplementary TCGA-2x2 leaking onto the public site.
    """
    index = (ROOT / "docs/results/index.rst").read_text()
    for slug in ("camelyon", "tcga-4x4", "tolkach-esca"):
        assert f"\n   {slug}\n" in index
        page = (ROOT / f"docs/results/{slug}.rst").read_text()
        assert f".. _{slug}:" in page
        assert f"results-table:: {slug}" in page
        assert f"themed-figure:: /_static/figures/pareto_{slug}" in page
        assert "tcga-2x2" not in page.lower()
    assert "tcga-2x2" not in index.lower()


def test_five_encoder_provenance_is_exact_and_auditable() -> None:
    provenance = (ROOT / "docs/results/index.rst").read_text()
    normalized = " ".join(provenance.split())
    expected_facts = [
        "e95e7ea15e039e78d74def101415e19d9a67ba80",  # Mascaret
        "e0ce6e0ee248470bd8604823e412ca64048a2495",  # Phaet
        "482d9519c6a10fc22fbe5bcd6a87d5daf056643c",  # RudolfV 2
        "b2cb55c8fff8aaaf9cc16fda6d09bfb21dfc6db8",  # RudolfV 2-B
        "76abacd512a98c72a6db6192af9fc98313c3bd78",  # RudolfV 2-S
        # Encoder, revision and batch as adjacent cells of the contract table.
        "Mascaret - ``e95e7ea15e039e78d74def101415e19d9a67ba80`` - 32",
        "Phaet - ``e0ce6e0ee248470bd8604823e412ca64048a2495`` - 64",
        "RudolfV 2 - ``482d9519c6a10fc22fbe5bcd6a87d5daf056643c`` - 32",
        "RudolfV 2-B - ``b2cb55c8fff8aaaf9cc16fda6d09bfb21dfc6db8`` - 32",
        "RudolfV 2-S - ``76abacd512a98c72a6db6192af9fc98313c3bd78`` - 64",
        "FP32",
        "checkpoint-native:model.encode",
        "concatenate-cls-and-mean-patches",
        "possible institutional/source-domain overlap",
        "does not establish leakage",
    ]

    for fact in expected_facts:
        assert fact in normalized


def test_final_report_covers_the_publication_decisions() -> None:
    report = (ROOT / "docs/publication-records/issue-133.md").read_text()
    normalized = " ".join(report.split())
    expected_facts = [
        "25 ranked pathology encoders plus DINOv2-B",
        "TCGA-2x2 remains supplementary/local",
        "Camelyon | 11",
        "TCGA-2x2 | 61",
        "TCGA-4x4 | 71",
        "Tolkach-ESCA | 61",
        "22,390 s",
        "approximately 28 minutes",
        "exact zero movement",
        "no sign changes",
        "no significance-threshold changes",
        "#124",
    ]

    for fact in expected_facts:
        assert fact in normalized


def test_primary_setup_docs_are_preserved_byte_faithfully() -> None:
    expected_sha256 = {
        "docs/agents/domain.md": "68d82c5d23c6e3834478246fadd3ac2c64cbc306a6795fc0d9c5778a9b513c5a",
        "docs/agents/issue-tracker.md": "b475e291673ac7d19e90c46a315d0816bfcf262c54820295d722255bcebc3f52",
        "docs/agents/triage-labels.md": "657fc9db24d2ab7044474e6cff75183e22ea78c5f29b433d2086065376c8835c",
    }
    glossary = """**Robustness-targeted fine-tune**:
A public encoder obtained by fine-tuning a base pathology encoder explicitly for
invariance to acquisition factors. It remains in the ranked panel, but its parent
relationship is reported so it is not mistaken for an independent pretraining run.
_Avoid_: robust model, new foundation model."""

    for relpath, expected in expected_sha256.items():
        assert hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest() == expected
    assert glossary in (ROOT / "CONTEXT.md").read_text()


def test_provenance_guard_accepts_the_expanded_family_relationships() -> None:
    rows = [
        ("Midnight-12k", 0.882, 0.559, 0.396),
        ("Mascaret", 0.876, 0.434, 0.270),
        ("RudolfV 2", 0.876, 0.562, 0.168),
        ("RudolfV 2-B", 0.866, 0.565, 0.169),
        ("H-optimus-1", 0.878, 0.676, 0.088),
        ("CONCHv1.5", 0.811, 0.492, 0.153),
        ("CONCH", 0.790, 0.487, 0.146),
        ("GenBio-PathFM", 0.851, 0.695, 0.155),
        ("DINOv2-B", 0.607, 0.579, 0.006),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["model", "bio_knn_bacc", "confounder_knn_bacc", "croma"],
    )

    macros = "\n".join(paper_values._provenance_macros(frame))

    assert r"\newcommand{\ProvenanceLeastConfModel}{Mascaret}" in macros
    assert r"\newcommand{\ProvenanceTwinModel}{CONCHv1.5}" in macros
    assert r"\newcommand{\ProvenanceFineTuneModel}{Mascaret}" in macros
    assert r"\newcommand{\ProvenanceFineTuneCroma}{$0.27$}" in macros


def test_historical_overlap_caption_uses_its_own_roster(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"model": "Midnight-12k", "rest": 0.60, "tcga": 0.82, "boost": 2.50},
            {"model": "Mascaret", "rest": 0.51, "tcga": 0.70, "boost": 1.80},
            {"model": "RudolfV 2", "rest": 0.41, "tcga": 0.50, "boost": 1.20},
            {"model": "DINOv2-B", "rest": 0.17, "tcga": 0.27, "boost": 1.21},
        ]
    )
    monkeypatch.setattr(overlap_table, "overlap_rows", lambda: frame)

    tex = overlap_table.build()

    assert "historical 3-encoder pathology subset" in tex
    assert r"$\dagger$ marks the $2$ TCGA-exposed encoders" in tex


def test_typed_neighbour_table_is_rendered_from_the_sealed_historical_roster(
    tmp_path: Path,
) -> None:
    archive = ROOT / typed_table.SNAPSHOT
    if not archive.is_dir():
        import pytest

        pytest.skip("sealed local paper-study archive is absent in this checkout")
    out = tmp_path / "typed_neighbor_ranks.tex"

    proc = subprocess.run(
        [sys.executable, str(REPRO / "generate_supp_rank_table.py"), "--out", str(out)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    tex = out.read_text()
    rows = {
        match.group(1)
        for match in re.finditer(r"^(.+?) & [-0-9.]+ & [0-9]+ & [0-9]+ \\\\$", tex, re.MULTILINE)
    }
    assert rows == EXPECTED_HISTORICAL_TYPED_PANEL
    assert "historical fixed 20-pathology-encoder analysis" in tex
    assert "% Sealed snapshot artifact SHA-256:" in tex
    assert "2facf7719b38ae7890d01dafe5b7b90c61b53a4409bba400c37a525d05fabc26" in tex
    assert (
        "first \\code{SO} or \\code{OS} neighbour occurs at a median rank of $\\approx 149$" in tex
    )


def test_no_unlabelled_stale_roster_claims_in_active_outputs() -> None:
    stale = re.compile(
        r"\b(?:20|21|22|23|twenty|twenty-one|twenty-two|twenty-three)"
        r"(?:[ -](?:tile-level[ -])?(?:model|encoder|pathology)|/2[123])",
        flags=re.IGNORECASE,
    )
    historical = re.compile(
        r"\b(?:historical|baseline|pre-existing|original|previous|prior|before|old)\b",
        flags=re.IGNORECASE,
    )
    active = [ROOT / "README.md", ROOT / "CONTEXT.md"]
    active.extend((ROOT / "docs").glob("*.rst"))
    active.extend((ROOT / "docs/results").glob("*.rst"))
    if PAPER.is_dir():
        seen: set[Path] = set()

        def add_reachable(path: Path) -> None:
            if path in seen or not path.exists():
                return
            seen.add(path)
            uncommented = "\n".join(
                re.sub(r"(?<!\\)%.*", "", line) for line in path.read_text().splitlines()
            )
            for match in re.findall(r"\\input\{([^}]+)\}", uncommented):
                child = PAPER / (match if match.endswith(".tex") else f"{match}.tex")
                add_reachable(child)

        for document in (PAPER / "main.tex", PAPER / "draft.tex"):
            add_reachable(document)
        active.extend(seen)

    violations: list[str] = []
    for path in active:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if stale.search(line) and not historical.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")

    assert violations == []


@pytest.mark.skipif(not PAPER.is_dir(), reason="paper/ is local and not present in clean checkouts")
def test_active_discussion_does_not_repeat_the_disproved_genbio_frontier_claim() -> None:
    discussion = (PAPER / "sections/discussion.tex").read_text()

    assert "GenBio-PathFM} was the only model on the median–tail Pareto frontier" not in discussion
    assert r"\TilePriorRankFrontierNModels" in discussion
    assert r"\TileRankFrontierNModels" in discussion


def _explicit_rank_pareto(
    models: list[str], median_ranks: list[int], tail_ranks: list[int]
) -> RankPareto:
    index = pd.Index(models, name="model")
    return RankPareto(
        medians=pd.DataFrame({"cohort": [0.0] * len(models)}, index=index),
        median_ranks=pd.DataFrame({"cohort": median_ranks}, index=index),
        tail_ranks=pd.DataFrame({"cohort": tail_ranks}, index=index),
        exposed=frozenset(),
        adversarial="cohort",
    )


def _explicit_frontier_panels() -> tuple[RankPareto, RankPareto]:
    historical = _explicit_rank_pareto(
        ["H1", "H2", "H3", "H4"],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    )
    current = _explicit_rank_pareto(
        [
            "Mascaret",
            "H1",
            "H2",
            "H3",
            "H4",
            "Phaet",
            "RudolfV 2",
            "RudolfV 2-B",
            "RudolfV 2-S",
        ],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 4, 3, 2, 6, 7, 8, 9],
    )
    return historical, current


def test_expanded_frontier_change_is_guarded_by_explicit_rank_inputs() -> None:
    historical, current = _explicit_frontier_panels()
    macros = paper_values._rank_frontier_change_macros(historical, current)

    assert r"\newcommand{\TilePriorRankFrontierNModels}{4}" in macros
    assert r"\newcommand{\TileRankFrontierNModels}{1}" in macros
    assert r"\newcommand{\TileRankFrontierModels}{Mascaret}" in macros


def test_frontier_guard_is_offline_at_the_generator_seam(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert not Path("output").exists()
    historical, current = _explicit_frontier_panels()

    macros = paper_values._rank_frontier_change_macros(historical, current)

    assert r"\newcommand{\TilePriorRankFrontierNModels}{4}" in macros
    assert r"\newcommand{\TileRankFrontierNModels}{1}" in macros
