"""Sphinx configuration for croma documentation."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from sphinx.deprecation import RemovedInSphinx10Warning

# sphinx-autodoc-typehints 3.6 still calls a Sphinx API deprecated for removal in 10.
# It is dependency noise rather than a documentation warning, and otherwise makes a
# successful ``sphinx -W`` build look non-clean on Sphinx 9.
warnings.filterwarnings(
    "ignore",
    category=RemovedInSphinx10Warning,
    module=r"sphinx_autodoc_typehints\._parser",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

from croma import __version__  # noqa: E402

project = "croma"
author = "Clément Grisi"
copyright = "2026, Clément Grisi"
release = __version__

# Consumed by the sidebar GitHub card (docs/_templates/sidebar/github.html): the version
# badge links to the release tag that matches the installed package.
html_context = {
    "github_latest_release_tag": release,
    "github_latest_release_url": (
        f"https://github.com/clemsgrs/croma/releases/tag/{release}"
    ),
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxext.opengraph",
    # Local: the themed-figure directive (docs/_ext/themedfigure.py), the directives that
    # render committed publication artifacts, and the inline role that computes
    # run-derived numbers from them (docs/_ext/resultsvalue.py).
    "themedfigure",
    "resultstable",
    "resultsvalue",
    "nipdtable",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Editor droppings: stale page copies that shadow real documents and break -W builds.
    ".ipynb_checkpoints",
    "**/.ipynb_checkpoints",
    # Internal decision records and working notes. They are Markdown, they are written for
    # contributors rather than users, and they read fine on GitHub -- so they stay out of
    # the published site instead of being converted.
    "adr",
    "reviewer-notes",
    # Figure sources (.tex) and their README: inputs to the committed figures, not pages.
    "figures",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    # The metric classes share a private BaseRobustnessIndex that is not part of the public
    # API and is not documented; surfacing it as a base would name something a reader
    # cannot look up.
    "show-inheritance": False,
    "special-members": False,
    "private-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# Link-preview metadata (LinkedIn, Slack, X). The per-page description is derived from
# the page's opening prose; the shared image is the dataset montage, the most legible
# single image of what the metrics measure.
ogp_site_url = "https://clemsgrs.github.io/croma/"
ogp_image = "_static/figures/dataset_montage.jpg"
ogp_image_alt = (
    "Pathology tiles arranged by biological class and medical center: the same class, "
    "rendered differently by different centers."
)
ogp_enable_meta_description = True

html_theme = "furo"
html_static_path = ["_static"]
# The published data tree, copied verbatim to the site root. It is what the distribution
# explorer fetches, and it makes every committed CSV directly downloadable from the site
# rather than only from the repository.
html_extra_path = ["../results"]
html_css_files = ["custom.css"]
# Loaded on every page but inert unless that page provides the corresponding mount point;
# each explorer fetches its committed publication payload only after finding a mount.
html_js_files = ["explorer.js", "nipd-explorer.js", "figures.js", "pareto.js"]
# Bare project name: the version lives in the sidebar card's release badge instead.
html_title = "croma"
html_show_sourcelink = False
# Furo's default sidebar, with the repository card inserted above the navigation tree.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/github.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}
html_theme_options = {
    "source_repository": "https://github.com/clemsgrs/croma",
    "source_branch": "main",
    "source_directory": "docs/",
}
