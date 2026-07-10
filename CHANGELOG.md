# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First public release of `croma`, a metrics-only library of representation-level
robustness metrics for pathology foundation models (RI, MaRI, and the flagship
cross-confounder margin metric).

Nothing in this project has been published to PyPI before, so everything below
describes the package as you first receive it. The bullets under *Changed* and
*Removed* record decisions taken during pre-release development; they are here
because they explain why the API looks the way it does, not because any released
version behaved differently.

### Added

- **Public API:** `RI`, `MaRI`, `CRoMa`, `expand_features_to_manifest`, and
  `__version__`. The metric classes are namespaces of classmethods; nothing is
  instantiated.
- **`croma` CLI** with `ri`, `mari`, `croma`, `build-embedding-manifest`, and
  `expand-embeddings` subcommands, each printing a JSON payload to stdout.
- **Documentation site** built with Sphinx and published to GitHub Pages, covering the
  metrics, the manifest and embedding contracts, the two evaluation designs, the CLI,
  and an autodoc API reference. Installable with the new `docs` extra. See
  `docs/adr/0009-docs-are-sphinx-on-github-pages.md`.
- **`RobustnessResult.tau`**, reporting the temperature MaRI actually scored with. `nan`
  on RI, which carries no temperature. `croma mari` additionally reports `tau_source`.
- **Design documentation under version control:** `CONTEXT.md` (glossary and casing map)
  and the architecture decision records in `docs/adr/`, so the reasoning behind the
  numbers and figures in the paper survives in the repository.

### Changed

- **MaRI's `tau` now defaults to automatic, per-dataset selection.** `MaRI.compute` and
  `MaRI.compute_curve` take `tau=None` by default, resolving it to the median typed
  (`SO`/`OS`) neighbour cosine distance at the operating `k`. Previously `tau` defaulted
  to a fixed `0.2`.

  This matters because MaRI weights neighbour evidence by `exp(-d / tau)`, so a
  principled `tau` sits on the scale of the typed-neighbour distances — and that scale is
  a property of each embedding. One fixed `tau` shared across models sharpens the margin
  for some and flattens it for others, which is the distortion MaRI exists to remove. The
  old default was off-scale enough to trip the package's own `warn_tau` check on the
  package's own toy fixture. The benchmark pipeline had always resolved `tau` per model;
  the library default now agrees with it.

  Selecting `tau` this way is not circular: `k` is chosen by biological kNN balanced
  accuracy, which never consults the weighting, so `k` is fixed before `tau` is chosen.
  An explicit `tau` is still honoured, and still warned about when off-scale.
- **The sdist is now an explicit include list** (`src`, `tests`, `docs`, `scripts`, and
  the metadata files) rather than "everything not ignored by git", so untracked scratch
  cannot reach PyPI.
- **Renamed the flagship metric CCMR → CRoMa.** The Cross-Confounder Margin Ratio
  (`CCMR`) is now the Cross-confounder Robustness Margin (`CRoMa`), and the library
  is named after it. This is a **name-only change, not a behaviour change**: the
  computation is identical and no reported value changes. "Ratio" was inaccurate —
  the metric is a signed, normalized margin in `(-1, 1)` (neutral at 0), not a
  ratio in `[0, ∞)` — and the new name shares "Robustness" with RI and MaRI.
  Casing disambiguates the homonym: `croma` (all-lowercase) is always the library,
  `CRoMa` (mixed case) is always the metric. This is a clean break with no `CCMR`
  compatibility shims. See `docs/adr/0001-rename-ccmr-to-croma.md`.
- **Renamed the optional extra `[bench]` → `[repro]`.** The extra that pulls in the
  paper-reproduction stack (data prep, embedding, plotting) is now installed with
  `pip install "croma[repro]"`. This is a name-only change; the dependency set is
  unchanged.
- **Adopted a src layout.** The package now lives under `src/croma/` so that the
  installed wheel — rather than the working tree — is exercised by tests and by
  `import croma`. Only the `croma` package ships in the wheel; `scripts/`, `tests/`,
  and `paper/` are excluded from the distribution.
- **Reset the version to `0.1.0`** for the first public, metrics-only release. The
  public API is the leanest surface: `RI`, `MaRI`, `CRoMa`,
  `expand_features_to_manifest`, and `__version__`.

### Removed

- **Dropped the `prune_ss_oo` and `summarize_by_mean` options** from `RI.compute`,
  `MaRI.compute`, and the benchmark driver (`--prune-ss-oo`, `--summarize-by-mean`),
  together with the code that existed only to serve them. Both were exploratory and
  have been abandoned. Neither was ever used to produce a committed result, so **no
  reported number changes**.
  - `summarize_by_mean` replaced a metric's headline `value` with the mean over the
    whole k-curve and `std` with the spread *across k*, while leaving every other
    column in the same row (tail metrics, `median`, undefined fractions, per-sample
    values) computed at a single `k = k_max`. The row mixed two incompatible
    estimators under one set of column names. `value` is now always the pooled score
    at the operating k, and `std` always the spread across pairs.
  - `prune_ss_oo` restricted neighborhoods to informative (SO/OS) neighbors, which
    drove `undefined_frac` to zero by construction and so destroyed the very
    diagnostics it was meant to sidestep. The SS/OO diagnostic columns
    (`undefined_frac`, `ss_dominated_undefined_frac`, `oo_dominated_undefined_frac`,
    `mixed_undefined_frac`) are **unchanged** and remain fully reported.
  - Both flags silently forced `selected_k = k_max`, overriding the requested k
    operating point. Removing them makes the k protocol (per-model `k-star` or the
    shared `median-k`) the sole determinant of the operating k.
  - Consequently removed: the `results/render_manifest.json` artifact (it carried
    only these two flags) and the four figures that only made sense under them
    (RI/MaRI cumulative-mean k-sweeps and RI/MaRI sample-distribution plots).
    `plot_croma_sample_distributions` is retained.

[0.1.0]: https://github.com/clemsgrs/croma/releases/tag/v0.1.0
