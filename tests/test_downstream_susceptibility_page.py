"""Rendered no-JavaScript contract for downstream shortcut susceptibility."""

from __future__ import annotations

import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag) -> None:
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data) -> None:
        if not self.hidden:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("nipd-docs")
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", "docs", str(destination)],
        cwd=ROOT,
        check=True,
    )
    return destination


def _text(path: Path) -> str:
    parser = _VisibleText()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.text


def test_homepage_and_navigation_distinguish_the_two_taxonomy_branches(rendered: Path) -> None:
    home = (rendered / "index.html").read_text(encoding="utf-8")
    text = _text(rendered / "index.html")
    assert "Representation robustness" in text
    assert "Downstream shortcut susceptibility" in text
    assert 'href="downstream-susceptibility.html"' in home


def test_page_explains_nipd_before_the_continuity_metric_and_public_api(rendered: Path) -> None:
    page = rendered / "downstream-susceptibility.html"
    text = _text(page)
    html = page.read_text(encoding="utf-8")
    assert "Negative nIPD means net degradation" in text
    assert "near zero means little or no net change over the confounding range" in text
    assert "positive nIPD means net improvement" in text
    assert "baseline skill" in text and "balanced accuracy minus chance" in text
    assert "predict the biological class" in text and "training composition" in text
    assert "ID is the primary mechanistic endpoint" in text
    assert "OOD also includes transfer effects" in text
    assert text.index("nIPD") < text.index("APD continuity")
    assert "from croma import nipd" in text
    assert "IAC" not in text
    assert 'href="nipd.json"' in html and 'href="nipd.csv"' in html


def test_page_mounts_an_accessible_interactive_evidence_browser(rendered: Path) -> None:
    page = rendered / "downstream-susceptibility.html"
    html = page.read_text(encoding="utf-8")
    text = _text(page)

    assert 'class="croma-nipd-explorer"' in html
    assert 'data-payload="nipd.json"' in html
    assert 'aria-label="Explore cohort-specific nIPD evidence"' in html
    assert "Interactive evidence browser" in text
    assert "Select a cohort first" in text
    assert "less degradation" in text
    assert "negative is net degradation" in text
    assert "zero is no net change" in text
    assert "positive is net improvement" in text
    assert "OOD also reflects transfer effects" in text
    assert "The static tables below are the complete no-JavaScript equivalent" in text
    assert 'src="_static/nipd-explorer.js' in html


def test_page_explains_the_model_level_croma_nipd_association(rendered: Path) -> None:
    page = rendered / "downstream-susceptibility.html"
    html = page.read_text(encoding="utf-8")
    text = _text(page)

    assert "CRoMa and downstream susceptibility" in text
    assert "median CRoMa at m=5" in text
    assert "associated with nIPD" in text
    assert "does not establish that CRoMa causes downstream performance" in text
    assert "different evaluation samples" in text
    assert "model-level comparison, not sample-level pairing" in text
    assert "ranked pathology encoders only" in text
    assert "DINOv2-B" in text and "excluded from the fitted trend and Spearman" in text
    assert "PCaBiop contains n=5 encoders" in text and "descriptive" in text
    assert 'src="_static/nipd-explorer.js' in html


def test_static_tables_cover_every_cohort_regime_without_javascript(rendered: Path) -> None:
    page = rendered / "downstream-susceptibility.html"
    html = page.read_text(encoding="utf-8")
    text = _text(page)
    for cohort, chance in (
        ("Camelyon", "0.500"),
        ("TCGA-4×4", "0.250"),
        ("Tolkach-ESCA", "0.167"),
        ("PCaBiop", "0.500"),
    ):
        assert cohort in text
        assert f"chance balanced accuracy: {chance}" in text
    assert text.count("Baseline balanced accuracy") == 8
    assert text.count("Baseline skill") >= 8
    assert text.count("Median CRoMa (m=5)") == 8
    assert "Spearman ρ = 0.94; n=25 ranked pathology encoders" in text
    assert "Spearman ρ = 0.60; n=5 ranked pathology encoders; descriptive" in text
    assert html.count('<table class="') >= 8
    assert "DINOv2-B †" in text
    assert (rendered / "nipd.json").read_bytes() == (ROOT / "results" / "nipd.json").read_bytes()
    assert (rendered / "nipd.csv").read_bytes() == (ROOT / "results" / "nipd.csv").read_bytes()
