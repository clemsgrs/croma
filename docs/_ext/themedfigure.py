"""A ``themed-figure`` directive: one figure, two theme variants, one caption.

furo renders light and dark, and the site's figures ship as a ``-light`` / ``-dark`` SVG
pair. Writing that by hand means two ``figure`` directives per figure with the caption
duplicated between them -- twice the source, and two places for a caption to drift.

::

    .. themed-figure:: _static/figures/croma_geometry
       :alt: CRoMa compares the distance to the nearest SO and OS neighbours.
       :width: 90%

       The margin CRoMa reports, for a fragile and a robust model.

emits both variants, tagged with furo's ``only-light`` / ``only-dark`` classes so exactly
one is visible. The argument is the path *stem* -- resolved as for any image directive --
without the theme suffix or extension.
"""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst.directives.images import Figure
from sphinx.application import Sphinx

THEMES = ("light", "dark")


class ThemedFigure(Figure):
    """Emit one ``figure`` node per theme, sharing this directive's caption."""

    def run(self) -> list[nodes.Node]:
        stem = self.arguments[0]
        emitted: list[nodes.Node] = []

        for theme in THEMES:
            # Figure.run() reads self.arguments[0], so point it at this theme's file and
            # let the base directive do the real work -- including caption parsing, which
            # re-parses self.content cleanly on each call.
            self.arguments[0] = f"{stem}-{theme}.svg"
            produced = super().run()

            figure = next((n for n in produced if isinstance(n, nodes.figure)), None)
            if figure is None:
                # Base directive failed (missing file, bad option); surface its messages
                # rather than swallowing them into a half-built pair.
                return produced

            figure["classes"].append(f"only-{theme}")
            emitted.append(figure)

        self.arguments[0] = stem
        return emitted


def setup(app: Sphinx) -> dict[str, object]:
    app.add_directive("themed-figure", ThemedFigure)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
