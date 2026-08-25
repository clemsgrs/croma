"""Acceptance coverage for the rendered model-request journey."""

from __future__ import annotations

import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[dict[str, str]] = []
        self._anchor: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._anchor = {key: value or "" for key, value in attrs}
            self._anchor["text"] = ""

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None:
            self._anchor["text"] = " ".join(self._anchor["text"].split())
            self.anchors.append(self._anchor)
            self._anchor = None


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._hidden_depth += 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._hidden_depth -= 1

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


@pytest.fixture(scope="module")
def rendered_docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("rendered-docs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "-b",
            "html",
            "docs",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    return output


def _anchors(path: Path) -> list[dict[str, str]]:
    parser = _AnchorParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.anchors


def _visible_text(path: Path) -> str:
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.text


def _assert_results_prompt(
    page: Path,
    *,
    explanation_end: str,
    prompt: str,
    form_template: str,
) -> None:
    html = page.read_text(encoding="utf-8")
    article = html.split('<article role="main"', maxsplit=1)[1]

    assert article.index(explanation_end) < article.index(prompt) < article.index("<table")
    prompt_link = next(anchor for anchor in _anchors(page) if anchor["text"] == prompt)
    assert prompt_link["href"] == (
        f"https://github.com/clemsgrs/croma/issues/new?template={form_template}"
    )


def test_sidebar_offers_model_evaluation_request_on_every_page(rendered_docs: Path) -> None:
    for page in ("index.html", "results/index.html"):
        html = (rendered_docs / page).read_text(encoding="utf-8")
        assert {
            "text": "Request a model evaluation",
            "href": ("" if page == "index.html" else "../") + "request-model.html",
        }.items() <= next(
            anchor.items()
            for anchor in _anchors(rendered_docs / page)
            if anchor["text"] == "Request a model evaluation"
        )

        sidebar = html.split('<aside class="croma-sidebar-github"', maxsplit=1)[1].split(
            "</aside>", maxsplit=1
        )[0]
        before_request = sidebar.split("Request a model evaluation", maxsplit=1)[0]
        assert "clemsgrs/croma" in before_request
        request_start = before_request.rsplit("</div>", maxsplit=1)[1].strip()
        assert request_start.startswith('<a class="croma-sidebar-github__request" href="')
        assert request_start.endswith('">')


def test_request_chooser_routes_public_tile_slide_and_private_models(rendered_docs: Path) -> None:
    page = rendered_docs / "request-model.html"
    anchors = _anchors(page)
    visible_text = _visible_text(page)

    assert (
        "evaluation on the three PathoROB tile cohorts: Camelyon, TCGA-4×4, and Tolkach-ESCA"
        in visible_text
    )
    assert {
        "text": "Request a tile encoder",
        "href": "https://github.com/clemsgrs/croma/issues/new?template=tile-encoder-request.yml",
    }.items() <= next(
        anchor.items() for anchor in anchors if anchor["text"] == "Request a tile encoder"
    )
    assert "evaluation on the PCaBiop whole-slide panel" in visible_text
    assert {
        "text": "Request a slide encoder",
        "href": "https://github.com/clemsgrs/croma/issues/new?template=slide-encoder-request.yml",
    }.items() <= next(
        anchor.items() for anchor in anchors if anchor["text"] == "Request a slide encoder"
    )
    private_link = next(
        anchor
        for anchor in anchors
        if anchor["text"]
        == "Private model? Email me to discuss whether an evaluation may be possible."
    )
    assert unquote(private_link["href"]) == (
        "mailto:clement.grisi@radboudumc.nl?subject=Private model evaluation inquiry"
    )


def test_tile_results_prompt_precedes_the_first_results_table(rendered_docs: Path) -> None:
    _assert_results_prompt(
        rendered_docs / "results" / "index.html",
        explanation_end="</p>",
        prompt="Don't see a tile encoder? Request an evaluation",
        form_template="tile-encoder-request.yml",
    )


def test_pcabiop_prompt_follows_the_explanation_and_precedes_the_first_results_table(
    rendered_docs: Path,
) -> None:
    _assert_results_prompt(
        rendered_docs / "results" / "pcabiop.html",
        explanation_end="row carries the † mark.",
        prompt="Don't see a slide encoder? Request an evaluation",
        form_template="slide-encoder-request.yml",
    )
