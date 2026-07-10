# 9. Documentation is Sphinx on GitHub Pages

Date: 2026-07-10

## Status

Accepted.

## Context

`croma` is about to take the `croma` name on PyPI. Ahead of that first release we owed a
decision on where the reference material lives.

The state we inherited was worse than "undecided". `README.md` carried a **Documentation**
section linking to `docs/metrics.md`, `docs/paired_evaluation.md`, `docs/benchmarking.md`
and `docs/croma-breakdown.md`. None of those files had ever existed — `git log` shows no
commit that added or removed any of them. The only Markdown under `docs/` outside `adr/`
was an untracked `documentation.md` describing `scripts/plotting.py` and
`scripts/benchmark.py` (both since moved or deleted) in pre-rename `CCMR` vocabulary.

So the real question was not "site or no site" but "the docs do not exist; in what form do
we write them?"

Three options:

1. **README-only.** Fold the embedding contract and the evaluation-design decision into
   `README.md`, delete the dead links. Smallest diff, nothing to maintain — and no home for
   the API reference, which is the material a library most needs.
2. **MkDocs Material.** Keep Markdown, publish it. Pleasant, but it gives no API autodoc,
   so every signature would be hand-transcribed and would rot. `croma`'s public surface is
   three classes whose *keyword defaults carry the methodology* (`tau`, `m`, `alpha`);
   those defaults must be generated from the source, not retyped.
3. **Sphinx.** Autodoc renders the real signatures and docstrings. It is also what
   [`slide2vec`](https://github.com/clemsgrs/slide2vec) already uses, so the toolchain,
   the `docs.yaml` workflow, and the furo theme are proven in a sibling repo by the same
   author.

## Decision

Sphinx, published to GitHub Pages by `.github/workflows/docs.yaml`, mirroring `slide2vec`:
furo, `sphinx.ext.autodoc`, `napoleon`, `intersphinx`, `sphinx-copybutton`,
`sphinx-autodoc-typehints`. Pages are `.rst`. Docs dependencies live in a `docs` extra.

The README shrinks to a landing page — what `croma` is, install, one quickstart per
interface — and points at the site for everything else.

Two consequences worth writing down:

- **`myst_parser` is deliberately not enabled.** `slide2vec` enables it; we have no
  Markdown page to render, and leaving it out keeps the ADRs from being pulled into the
  site as orphan documents.
- **`docs/adr/` and `docs/reviewer-notes/` are excluded from the build.** They are written
  for contributors, they read fine on GitHub, and converting them to `.rst` would be churn
  for no reader. `exclude_patterns` in `conf.py` enforces this.

The build runs with `-W` (warnings as errors) on every pull request, so a dead
cross-reference or an orphaned page fails CI instead of shipping.

## Consequences

GitHub Pages must be enabled for the repository with **Source: GitHub Actions** before the
first `deploy` job can succeed; the `build` job passes regardless. Until it is enabled,
`docs.yaml` will build on PRs and fail only at the deploy step on `main`.

Autodoc does not follow module-level aliases: `.. autoclass:: croma.MaRI` renders as
"alias of MarginAwareRobustnessIndex" with no members. `docs/api.rst` therefore documents
the canonical classes (`croma.MarginAwareRobustnessIndex`, …) and maps the short import
names to them in a table.

Adding a public method now means adding it to `docs/api.rst`, or it goes undocumented.
That is the intended pressure.
