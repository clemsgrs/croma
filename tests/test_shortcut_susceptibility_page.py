"""Rendered contract for the shortcut susceptibility publication surfaces."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

COHORT_PAGES = {
    "camelyon": "camelyon.html",
    "tcga-4x4": "tcga-4x4.html",
    "tolkach-esca": "tolkach-esca.html",
    "pcabiop": "pcabiop.html",
}


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


def _payload() -> dict:
    return json.loads((ROOT / "results" / "nipd.json").read_text(encoding="utf-8"))


def _nipd_table_rows(html: str, caption_fragment: str) -> list[str]:
    """First-column model names of the rendered table whose caption matches."""
    for chunk in html.split("<table")[1:]:
        caption = re.search(r"<caption[^>]*>(.*?)</caption>", chunk, flags=re.S)
        if caption is None or caption_fragment not in re.sub(r"<[^>]+>", "", caption.group(1)):
            continue
        cells = re.findall(r"<tr[^>]*>\s*<td><p>([^<]+)", chunk)
        return [cell.strip() for cell in cells]
    raise AssertionError(f"no rendered table with caption containing {caption_fragment!r}")


def test_homepage_and_navigation_distinguish_the_two_taxonomy_branches(rendered: Path) -> None:
    home = (rendered / "index.html").read_text(encoding="utf-8")
    text = _text(rendered / "index.html")
    assert "Representation robustness" in text
    assert "Shortcut susceptibility" in text
    assert "Downstream shortcut susceptibility" not in text
    assert 'href="shortcut-susceptibility.html"' in home


def test_request_page_is_reachable_through_the_sidebar_button_only(rendered: Path) -> None:
    home = (rendered / "index.html").read_text(encoding="utf-8")
    assert (rendered / "request-model.html").exists()
    assert "croma-sidebar-github__request" in home
    sidebar = re.search(r'<div class="sidebar-tree">.*?</div>', home, flags=re.S)
    assert sidebar is not None and "request-model" not in sidebar.group(0)


def test_old_url_redirects_to_the_renamed_page(rendered: Path) -> None:
    stub = (rendered / "downstream-susceptibility.html").read_text(encoding="utf-8")
    assert 'http-equiv="refresh"' in stub
    assert "url=shortcut-susceptibility.html" in stub


def test_page_explains_nipd_before_the_continuity_metric_and_public_api(rendered: Path) -> None:
    page = rendered / "shortcut-susceptibility.html"
    text = _text(page)
    html = page.read_text(encoding="utf-8")
    assert "Negative nIPD means net degradation" in text
    assert "positive nIPD means net improvement" in text
    assert "can hide offsetting changes" in text
    assert "predicts the biological class" in text and "training composition" in text
    assert "margin over chance" in text
    assert "Baseline skill" not in text and "baseline skill" not in text
    assert "ID is the primary mechanistic endpoint" in text
    assert "OOD also includes transfer effects" in text
    assert text.index("nIPD") < text.index("APD")
    assert "IAC" not in text
    assert 'href="nipd.json"' in html and 'href="nipd.csv"' in html


def test_page_mounts_an_accessible_interactive_evidence_browser(rendered: Path) -> None:
    page = rendered / "shortcut-susceptibility.html"
    html = page.read_text(encoding="utf-8")
    text = _text(page)

    assert 'class="croma-nipd-explorer"' in html
    assert 'data-payload="nipd.json"' in html
    assert 'aria-label="Explore cohort-specific nIPD evidence"' in html
    assert "Interactive evidence browser" in text
    assert 'src="_static/nipd-explorer.js' in html
    for cohort_page in COHORT_PAGES.values():
        assert f'href="results/{cohort_page}"' in html


def test_page_explains_the_model_level_croma_nipd_association(rendered: Path) -> None:
    page = rendered / "shortcut-susceptibility.html"
    text = _text(page)

    assert "CRoMa and downstream susceptibility" in text
    assert "median CRoMa at m=5" in text
    assert "does not establish that CRoMa causes downstream performance" in text
    assert "model-level comparison, not sample-level pairing" in text
    assert "ranked pathology encoders only" in text
    assert "DINOv2-B" in text and "excluded from the fitted trend and Spearman" in text
    assert "n=5" in text and "descriptive" in text


def test_cohort_pages_carry_the_static_tables_sorted_by_decreasing_nipd(rendered: Path) -> None:
    payload = _payload()
    for cohort in payload["cohorts"]:
        page = rendered / "results" / COHORT_PAGES[cohort["slug"]]
        html = page.read_text(encoding="utf-8")
        text = _text(page)
        assert "Shortcut susceptibility" in text
        assert f"chance balanced accuracy: {cohort['chance']:.3f}" in text
        assert text.count("Median CRoMa (m=5)") == 2
        assert text.count("Baseline balanced accuracy") == 2
        assert "Baseline skill" not in text
        for regime, heading in (("id", "ID"), ("ood", "OOD")):
            ranked = sorted(
                (model for model in cohort["models"] if not model["is_control"]),
                key=lambda model: -model["regimes"][regime]["nipd"],
            )
            control = [model for model in cohort["models"] if model["is_control"]]
            expected = [model["model"] for model in ranked] + [
                f"{model['model']} †" for model in control
            ]
            rows = _nipd_table_rows(html, f"{cohort['label']} — {heading}; Spearman")
            assert rows == expected, f"{cohort['slug']}/{regime} table order"
    pcabiop_text = _text(rendered / "results" / COHORT_PAGES["pcabiop"])
    assert "MOOZY" in pcabiop_text and "PAR" in pcabiop_text

    assert (rendered / "nipd.json").read_bytes() == (ROOT / "results" / "nipd.json").read_bytes()
    assert (rendered / "nipd.csv").read_bytes() == (ROOT / "results" / "nipd.csv").read_bytes()
