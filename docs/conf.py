"""Sphinx configuration for croma documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from croma import __version__  # noqa: E402

project = "croma"
author = "Clément Grisi"
copyright = "2026, Clément Grisi"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Internal decision records and working notes. They are Markdown, they are written for
    # contributors rather than users, and they read fine on GitHub -- so they stay out of
    # the published site instead of being converted.
    "adr",
    "reviewer-notes",
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

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"croma {release}"
html_show_sourcelink = False
html_theme_options = {
    "source_repository": "https://github.com/clemsgrs/croma",
    "source_branch": "main",
    "source_directory": "docs/",
}
