# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First public release of `croma`, a metrics-only library of representation-level
robustness metrics for pathology foundation models (RI, MaRI, and the flagship
cross-confounder margin metric).

### Changed

- **Renamed the flagship metric CCMR → CRoMa.** The Cross-Confounder Margin Ratio
  (`CCMR`) is now the Cross-confounder Robustness Margin (`CRoMa`), and the library
  is named after it. This is a **name-only change, not a behaviour change**: the
  computation is identical and no reported value changes. "Ratio" was inaccurate —
  the metric is a signed, normalized margin in `(-1, 1)` (neutral at 0), not a
  ratio in `[0, ∞)` — and the new name shares "Robustness" with RI and MaRI.
  Casing disambiguates the homonym: `croma` (all-lowercase) is always the library,
  `CRoMa` (mixed case) is always the metric. This is a clean break with no `CCMR`
  compatibility shims. See `docs/adr/0001-rename-ccmr-to-croma.md`.
- **Widened the one-time `ccmr` → `croma` migration script** beyond its original
  header-only design. In addition to renaming legacy `ccmr*` CSV column headers,
  `scripts/experiments/migrate_ccmr_columns.py` now also rewrites the `ccmr`
  identifier where it survives as a **cell value** (metric labels like `ccmr` /
  `ccmr_m5` and comparison keys like `ccmr_vs_ri`, using the same leading-token
  rule) and renames affected CSV **filenames** (e.g.
  `model_specific_ccmr_subgroups.csv` → `model_specific_croma_subgroups.csv`).
  This remains a pure identifier change with **no recompute** — every number is
  preserved, and cells that merely contain `ccmr` as a substring (e.g. paths) are
  left byte-for-byte unchanged. The migration stays idempotent. Accepted as a
  documented exception in `docs/adr/0001-rename-ccmr-to-croma.md`.
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

### Added

- Tracked design documentation under version control: `CONTEXT.md` (glossary and
  casing map), the architecture decision records in `docs/adr/`, and the
  paper-reproduction/experiment scripts under `scripts/experiments/`, so the
  numbers and figures in the paper are reproducible from a clean checkout.

[0.1.0]: https://github.com/clemsgrs/croma/releases/tag/v0.1.0
