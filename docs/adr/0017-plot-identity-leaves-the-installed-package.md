# The plot identity leaves the installed package

`croma.plotstyle` and its bundled Arimo typefaces move out of `src/croma/` and into the
benchmark plotting library, as `plotting.style` and `plotting/fonts/`. Nothing about the
shared visual identity changes; only where it is installed from.

## Why

The first release made the cost visible. `croma-0.1.0-py3-none-any.whl` came to 2.1 MB, of
which **1.93 MB was four font files** — a metrics library whose stated core dependencies
are `numpy`, `pandas`, `scikit-learn` and `tqdm` was shipping a typeface family to every
installer, for a plotting convenience that cannot run at all without `matplotlib`, which
lives in the `repro` extra.

The module was already in the wrong place, and the import graph said so plainly. Of the 35
files that imported `croma.plotstyle`, **every one was under `scripts/`** — the benchmark
plot archetypes, the paper figure generators, the studies, the dev tools. Nothing in
`src/croma/` imported it, and it was never exported from `croma/__init__.py`. It was part
of the reproduction layer that happened to be installed alongside the library.

That makes this a correction to ADR-0002 rather than an exception to it. ADR-0002 scoped
the library to the metrics; ADR-0011 later widened it to `croma.downstream`, on the reasoning
that APD/nAPD are *named contributions of the paper* and a contribution that ships in no
package cannot be reused. The plot identity is not a contribution. It is how the figures
look.

**Gating by an extra was considered and is not possible.** Extras add dependencies, not
modules: `pip install croma` and `pip install croma[repro]` install the same files. The two
real options were a second distribution (`croma-plotstyle`, listed in `repro`) or moving the
module out of the installed package entirely. A second PyPI name, version and release
pipeline is a large standing cost for a module only the un-shipped `scripts/` tree consumes
— and `scripts/` is not in the wheel at all, so a repro user is working from a clone and
already has the file.

**The timing is the cheap part.** `croma` had never been published when this was decided, so
a breaking move of a public import path broke nobody. After 0.1.0 it would have been a
breaking change for real installs.

## Shape

- **`scripts/bench/plotting/style.py`** — the former `croma/plotstyle.py`, unchanged but for
  its docstring.
- **`scripts/bench/plotting/fonts/`** — the four Arimo faces, unchanged.
- **Call sites inside the plotting package** use relative imports (`from .style import ...`).
- **Call sites elsewhere** (`scripts/repro/`, `scripts/studies/`, `scripts/tools/`, the
  local-only paper tests) reach it as `from plotting.style import ...`, with
  `scripts/bench` placed on `sys.path` by the same shim idiom those files already use for
  `src/` (ADR-0006 accepts cross-directory imports in the reproduction layer).

## Consequences

- **The core wheel drops from 2.1 MB to ~180 KB**, and the README's "depends only on numpy,
  pandas, scikit-learn and tqdm" is now true of what is installed, not merely of what is
  imported.
- **`import croma.plotstyle` no longer resolves.** There is no deprecation shim, because
  there is no released version in which it worked.
- **A figure cannot be rendered from an installed `croma` alone.** It never could — every
  renderer is in `scripts/` — but the dependency is now structural instead of implied.
- **`scripts/bench/` becomes load-bearing for the paper figures**, which live in the
  local-only `scripts/repro/` (ADR-0012). The tracked half now owns the identity the
  untracked half depends on. That is the right way round: the published deliverable owns
  the shared code.

## When this reverses

If the plot identity ever becomes something an outside user should apply to their own
figures — a stated deliverable rather than an internal consistency mechanism — it earns a
place in the package on ADR-0011's reasoning, and the fonts can be subset to keep the
weight honest.
