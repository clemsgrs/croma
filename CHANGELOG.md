# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING: the manifest's required `slide_id` column is now `group_id`.** The field
  never meant "slide": it names the *independence group* a sample belongs to — a slide, a
  patient, a specimen, an acquisition — and candidates sharing a query's value are excluded
  before neighbours are selected. `slide_id` made a statistical contract read as a
  pathology-specific one. There is no alias, no deprecation period and no configurable
  group column: a manifest carrying only `slide_id` fails naming the missing `group_id`,
  and a blank or missing `group_id` fails rather than passing as a group of its own. If
  both columns are present only `group_id` has metric semantics. Rename the column in every
  manifest (`csv` header and preparation script), and re-run scoring: `group_id` is part of
  the manifest fingerprint and the embedding-alignment key, so changing a row's value
  invalidates score-dependent cached artifacts. Feature embeddings stay reusable, since
  their identity is the tiles they were extracted from.
  Per-sample benchmark artifacts now carry a `group_id` column.

### Added

- **`croma.nipd`**, normalized integrated performance degradation. It divides the
  repeat-mean degradation curve by above-chance baseline skill and integrates it over
  Cramér's \(V\) by the trapezoidal rule. This corrects unequal baseline headroom while
  preventing the density of a benchmark's sampled splits from defining their weight.
  The supplied Cramér's-\(V\) coordinates must span 0 to 1, and mean baseline balanced
  accuracy must exceed chance; there is no additional weak-skill gate. See ADR-0018.
- **`results/`, a tracked publication artifact.** A small set of CSVs, a binned
  distribution payload and a provenance sidecar, written from the benchmark runs by
  `scripts/tools/export_results.py`. The documentation site renders them, and the README's
  results table is a generated region of the file rather than a hand-typed one.
  `output/` is unchanged: still git-ignored, still safe to wipe. See ADR-0016.
- **A results section on the documentation site** — a cross-cohort aggregate with two
  independent rank columns and no combined score, per-cohort pages carrying the full column
  set, and a distribution explorer that counts the samples in any range you drag over.
- **`tests/test_results_export.py`**, which regenerates `results/` from `output/` and fails
  on any difference, so a benchmark re-run that was never republished is caught rather than
  silently leaving stale numbers on a public site. It skips where `output/` is absent.

### Removed

- **`croma.napd`**, the equal-weight average of chance-normalized degradation over sampled
  splits. It is replaced by `croma.nipd`, whose trapezoidal integral targets a uniform
  average over Cramér's \(V\) instead of the empirical density of a benchmark's grid.
  No compatibility alias is retained because the two names denote different estimands.

### Fixed

- **Dark-mode figures.** The page background is now stripped before the dark pass rather
  than after — the strip matched on white, and the dark pass had already recoloured it, so
  every plotted figure sat in a visible grey box. And a colour now inverts when it is
  merely very pale, whatever its hue: nothing legible is drawn that pale, so a pale tint is
  always a background wash.

## [0.1.0] - 2026-07-28

First public release of `croma`, a lean library of robustness metrics for pathology
foundation models: the representation-level metrics (RI, MaRI, and the flagship
cross-confounder margin metric) plus `croma.downstream`, which measures shortcut
susceptibility on a downstream task (the confounder-biased probe protocol and its two
reductions, APD and nAPD).

`croma.downstream` is numpy/sklearn-only and adds no install weight; `torch` remains
confined to the `[repro]` extra. See ADR-0011.

This is the first release under the name `croma`. Everything below describes the
package as you first receive it. The bullets under *Changed* and *Removed* record
decisions taken during pre-release development; they are here because they explain
why the API looks the way it does, not because any released version behaved
differently.

> The project was briefly distributed as `cross-margin` 1.0.0 (March 2026), before the
> metric was renamed and before `tau` was resolved automatically. That name is retired;
> `croma` is the only supported distribution. There is no upgrade path from it, and no
> `cross-margin` code is carried forward.

### Added

- **Public API:** `RI`, `MaRI`, `CRoMa`, `probe_sweep`, `apd`, `napd`,
  `expand_features_to_manifest`, and `__version__`. The metric classes are namespaces of
  classmethods; nothing is instantiated.
- **`croma.downstream.probe_sweep`**, the confounder-biased probe protocol: it trains a
  biology probe on frozen embeddings while a schedule walks the training set from balanced
  to fully confounded, scores each probe on test rows that do not move, and returns the
  `(n_splits, n_iterations)` matrix `apd` and `napd` reduce — untouched, with no reshaping.
  It consumes embeddings and a split assignment only: no model is loaded, no manifest read
  and no output layout referenced, which is the boundary ADR-0011's narrowing of ADR-0002
  rests on. Two further PathoROB functions are vendored verbatim for it under the same
  terms as the APD reduction — the split-mapping helper and the logistic-probe trainer —
  and `tests/fixtures/pathorob_schedule_parity.json` holds the schedules upstream's own
  helper produces, which croma must reproduce exactly. `croma.downstream` also exposes
  `probe_sweep_over_test_sets`, which scores an unseen confounder off the same training
  pass, `pathorob_schedule`, which builds PathoROB's own schedules, and `IN_DOMAIN`, the
  key the sweep's own held-out matrix comes back under; none is promoted to the top level,
  so none carries a stability promise. `probe_sweep_over_test_sets` additionally takes
  `arrange_slides`, replacing the one step of a replicate that decides which slides a split
  trains on and which sit in the held-out tail — for a cohort whose slides cannot be ordered
  freely, such as PathoROB's Tolkach-ESCA and its per-replicate case split. Its default is
  the sweep's own shuffle, so the reference protocol is unchanged; see ADR-0015. The
  protocol is numpy/sklearn only and adds no install weight.
- **`croma.downstream.apd`**, PathoROB's Average Performance Drop, reducing the same
  `(n_splits, n_iterations)` accuracy matrix as `napd` but against raw accuracy and with
  no `chance` argument. It is reported as the *faithful reference*, so its reduction is
  vendored verbatim from PathoROB (BSD 3-Clause, © 2025 BIFOLD Pathomics) rather than
  reimplemented — vendored rather than depended on, because PathoROB is not on PyPI and
  PyPI rejects direct-URL dependencies. The vendored code is frozen: upstream changes
  deliberately do not propagate. `tests/fixtures/pathorob_apd_parity.json` pairs accuracy
  matrices PathoROB published with the APD values PathoROB published for them, and the
  reduction must reproduce them exactly; the fixture is hermetic, so it runs in CI with no
  PathoROB checkout and no skip path. Attribution ships in the distribution as `NOTICE`.
  Note `apd` keeps PathoROB's mean-of-ratios order while `napd` uses ratio-of-means; the
  two are deliberately not aligned. See ADR-0011 and ADR-0014.
- **`croma.downstream.napd`**, the skill-normalized Average Performance Drop, reducing
  an `(n_splits, n_iterations)` matrix of balanced accuracies against an explicit
  `chance = 1 / n_biological_classes`. Skill is accuracy above chance, so the value is
  the share of *learnable* signal the confounder destroys and is comparable across tasks
  with different class counts. It averages the replicate axis before taking the ratio,
  unlike APD — the one deliberate deviation from PathoROB's reduction, recorded in
  `docs/adr/0014-napd-averages-replicates-before-taking-the-ratio.md`. It carries no
  skill floor: every admissible input yields a value.
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

- **`croma.plotstyle` and its bundled typefaces are not part of the package.** The shared
  plot identity now lives with the benchmark plotting library as `plotting.style`
  (`scripts/bench/plotting/`). It was only ever imported from `scripts/` — nothing in the
  metrics library used it, and it was never exported from `croma/__init__.py` — while its
  four Arimo faces accounted for 1.93 MB of a 2.1 MB wheel, paid by every installer for a
  convenience that needs `matplotlib` from the `[repro]` extra to do anything. The wheel is
  now 63 KB, and "depends only on `numpy`, `pandas`, `scikit-learn` and `tqdm`" describes
  what is installed, not just what is imported. Rendering was always driven from `scripts/`,
  so no figure changes. See ADR-0017.

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

### Fixed

- **The reproduction study's Tolkach-ESCA case arrangement is deterministic.** It selected
  a replicate's held-out cases by iterating `set(test_cases) & set(cases)`, and slides land
  in the held-out tail in the order they are iterated — so which cases sat outermost
  followed CPython's per-process string hashing. The tail is narrower than the number of
  cases a replicate draws, so this decided the test set's *membership*, not just its order,
  and Tolkach's stored accuracy matrices were never reproducible across processes. The
  intersection is now sorted.

  This is a **study-layer** fix: nothing in the distributed package changes, since
  `scripts/` is not shipped. It is recorded here because it bears on the reproducibility of
  published numbers. The behaviour was inherited verbatim from PathoROB's own driver, so
  sorting is a deliberate divergence from the reference protocol on one of the three
  faithful benchmarks, and **every Tolkach number moves once the sweep is re-run**. The
  committed Tolkach matrices predate the fix and are not regenerated here. See #105.

[0.1.0]: https://github.com/clemsgrs/croma/releases/tag/v0.1.0
