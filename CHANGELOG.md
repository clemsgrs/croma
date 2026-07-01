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
  is named after it. This is a **name change, not a behaviour change**: the
  computation is identical and no reported value changes. "Ratio" was inaccurate —
  the metric is a signed, normalized margin in `(-1, 1)` (neutral at 0), not a
  ratio in `[0, ∞)` — and the new name shares "Robustness" with RI and MaRI.
  Casing disambiguates the homonym: `croma` (all-lowercase) is always the library,
  `CRoMa` (mixed case) is always the metric. This is a clean break with no `CCMR`
  compatibility shims. See `docs/adr/0001-rename-ccmr-to-croma.md`.

### Added

- Tracked design documentation under version control: `CONTEXT.md` (glossary and
  casing map), the architecture decision records in `docs/adr/`, and the
  paper-reproduction/experiment scripts under `scripts/experiments/`, so the
  numbers and figures in the paper are reproducible from a clean checkout.

[0.1.0]: https://github.com/clemsgrs/croma/releases/tag/v0.1.0
