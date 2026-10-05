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
    assert re.search(
        r"<strong>nIPD measures\s+the share of above-chance performance that is lost</strong>",
        html,
    )
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
    assert "normalized change at V = 1" in text
    assert "\u2248 chance" in text
    assert 'src="_static/nipd-explorer.js' in html
    for cohort_page in COHORT_PAGES.values():
        assert f'href="results/{cohort_page}"' in html


def test_page_explains_the_model_level_croma_nipd_association(rendered: Path) -> None:
    page = rendered / "shortcut-susceptibility.html"
    text = _text(page)

    assert "CRoMa and downstream susceptibility" in text
    assert "median CRoMa at m=5" in text
    assert "ranked pathology encoders only" in text
    assert "DINOv2-B is excluded from both" in text
    assert "n=5" in text and "descriptive" in text


def test_cohort_pages_carry_the_static_tables_ranked_by_the_endpoint(rendered: Path) -> None:
    payload = _payload()
    for cohort in payload["cohorts"]:
        page = rendered / "results" / COHORT_PAGES[cohort["slug"]]
        html = page.read_text(encoding="utf-8")
        text = _text(page)
        assert "Shortcut susceptibility" in text
        assert "chance balanced accuracy" not in text
        assert "Each caption reports Spearman \u03c1" in text
        assert text.count("Median CRoMa (m=5)") == 2
        assert text.count("Change at V = 1") == 3  # both captions' column, plus the prose
        assert text.count("Baseline balanced accuracy") == 2
        assert "Baseline skill" not in text
        for regime, heading in (("id", "ID"), ("ood", "OOD")):
            # The endpoint, not the pooled area: an early gain must not pay for a
            # late collapse in the ordering the reader sees first.
            ranked = sorted(
                (model for model in cohort["models"] if not model["is_control"]),
                key=lambda model: -model["regimes"][regime]["mean_normalized_trajectory"][-1],
            )
            control = [model for model in cohort["models"] if model["is_control"]]
            expected = [model["model"] for model in ranked] + [
                f"{model['model']} †" for model in control
            ]
            rows = _nipd_table_rows(html, f"{cohort['label']} — {heading}; Spearman")
            assert rows == expected, f"{cohort['slug']}/{regime} table order"
    pcabiop_text = _text(rendered / "results" / COHORT_PAGES["pcabiop"])
    assert "MOOZY" in pcabiop_text and "PAR" in pcabiop_text

    # The endpoint column names the encoders whose curve reaches chance, which the
    # pooled area hides: Phikon-v2 holds an nIPD of -0.199 and still ends at -0.971.
    camelyon_text = _text(rendered / "results" / COHORT_PAGES["camelyon"])
    assert "-0.971 \u2248 chance" in camelyon_text
    assert camelyon_text.count("\u2248 chance") == 4  # the prose, plus three ID rows
    # No Tolkach-ESCA curve comes near chance, so only the prose names the flag.
    assert _text(rendered / "results" / COHORT_PAGES["tolkach-esca"]).count("\u2248 chance") == 1

    assert (rendered / "nipd.json").read_bytes() == (ROOT / "results" / "nipd.json").read_bytes()
    assert (rendered / "nipd.csv").read_bytes() == (ROOT / "results" / "nipd.csv").read_bytes()


def test_tables_bold_the_leader_of_every_higher_is_better_column(rendered: Path) -> None:
    """PCaBiop OOD is the case the bolding exists for: nIPD leads on one encoder while
    CRoMa and the endpoint lead on another, because MOOZY's early gain and late collapse
    cancel inside the pooled area. The table is ranked on the endpoint, so PRISM2 leads
    it and MOOZY keeps only the bold nIPD that motivated the change."""
    html = (rendered / "results" / COHORT_PAGES["pcabiop"]).read_text(encoding="utf-8")
    table = next(
        chunk
        for chunk in html.split("<table")[1:]
        if "PCaBiop — OOD; Spearman" in re.sub(r"<[^>]+>", "", chunk[: chunk.index("</caption>")])
    )
    rows = {
        re.search(r"<td><p>([^<]+)", row).group(1).strip(): row.split("</td>")
        for row in table.split("<tr")[2:]
    }
    assert "<strong>0.048</strong>" in rows["MOOZY"][3]
    assert "<strong>0.270</strong>" in rows["PRISM2"][1]
    assert "<strong>-0.040</strong>" in rows["PRISM2"][2]
    assert list(rows)[0] == "PRISM2"
    # A higher baseline is not a better one, so that column has no leader.
    assert "<strong>" not in "".join(row[4] for row in rows.values())
    # The unranked control never carries a mark, whatever its values.
    camelyon = (rendered / "results" / COHORT_PAGES["camelyon"]).read_text(encoding="utf-8")
    for row in camelyon.split("<tr")[1:]:
        row = row.split("</tr>")[0]
        if "†" in row.split("</td>")[0]:
            assert "<strong>" not in row
