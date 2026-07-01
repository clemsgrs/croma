# Rename the flagship metric CCMR → CRoMa

We renamed the Cross-Confounder Margin Ratio (**CCMR**) to the Cross-confounder
Robustness Margin (**CRoMa**), styled mixed-case, and made it the metric the
`croma` library is named after.

## Why

- **"Ratio" was inaccurate.** The metric was redefined during development as a
  signed, normalized margin in `(-1, 1)` (neutral at 0), not a ratio in `[0, ∞)`.
  "Robustness Margin" describes what it actually computes.
- **Cohesion.** RI (Robustness Index), MaRI (Margin-aware Robustness Index), and
  CRoMa now share "Robustness"; CRoMa's mixed case mirrors MaRI.
- **Branding.** The paper's contribution is the metric; naming the library after
  its flagship metric is deliberate.

## The homonym, and how we resolve it

`croma` (the library) and `CRoMa` (the metric) are homonyms. We accepted this
("flagship model": the library is named after its headline metric) rather than
rename the package or coin a non-colliding metric name. Casing carries the
disambiguation:

- `croma` — all-lowercase, code font — is always the library.
- `CRoMa` — mixed case — is always the metric.

A per-context casing map (see `CONTEXT.md` and the rename PR) renders the token
consistently: `CRoMa` in prose/docstrings and the public API alias; CapWords
`CrossConfounderRobustnessMargin` for the class; `CRoMaResult` for the result
type; `CROMA_HEADLINE_M` for constants; lowercase `croma` for modules, the
`--croma-*` CLI flags, and `croma*` CSV columns; `Croma` for LaTeX macro ids.

## Consequences

- **Clean break, no shims.** Nothing is published under `croma` yet, so no `CCMR`
  Python alias and no dual-read of `ccmr`/`croma` CSV columns. The code only ever
  knows `croma`. Announced in the CHANGELOG. Internal consumers (paper pipeline,
  reproduction notebooks) update one import and one column name.
- **Existing CSVs are migrated, not regenerated.** The rename does not change any
  value, so a one-time throwaway migration script renames `ccmr*` → `croma*`
  columns across the `output/` CSVs (no pipeline re-run, no metric recompute). The
  paper's generated tables/macros are then re-emitted from the migrated CSVs. A
  one-time migration is not a back-compat shim — nothing in the shipped code reads
  the old names.
- **The one-time migration was extended (accepted exception).** The identifier
  `ccmr` survives beyond column headers: it also appears as **cell values** in a
  few analysis CSVs (metric labels such as `ccmr` / `ccmr_m5`, and comparison keys
  such as `ccmr_vs_ri` / `ccmr_vs_mari`, read by the already-`croma`-aware value
  and table generators) and in a few CSV **filenames** (e.g.
  `model_specific_ccmr_subgroups.csv`). The originally header-only migration script
  (`scripts/experiments/migrate_ccmr_columns.py`) was therefore widened, as an
  accepted one-time exception, to also rewrite the metric identifier in those known
  label/key cell values (same conservative leading-token rule as the header pass —
  only whole-cell `ccmr` or `ccmr_<suffix>` matches; substrings such as paths are
  left untouched) and to rename the affected CSV filenames (`ccmr` → `croma` token
  in the basename). This is **still no recompute**: every stored value is preserved
  byte-for-byte and the migration remains idempotent. It is not a back-compat shim
  — the shipped code still only knows `croma`.
