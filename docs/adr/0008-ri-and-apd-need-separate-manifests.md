# 8. RI and APD need separate manifests per cohort

Date: 2026-07-09

## Status

Accepted. Extends [ADR-0007](0007-embeddings-are-a-tileset-benchmarks-are-views.md).

## Context

PathoROB evaluates the Robustness Index and the Adversarial Performance Drop on
*different row sets* of the same cohort. We had one manifest per cohort plus an ad-hoc
`-faithful.csv`, and an `id_ood` column that looked authoritative. It was not, and the
ambiguity had already produced a silent defect.

Reading PathoROB's own source:

- **RI** (`pathorob/robustness_index/robustness_index.py`, `get_meta`) loads a metadata
  file, then drops `subset == "OOD"`.
- **APD** (`pathorob/apd/utils.py`) loads a *different* metadata file and uses both the
  `ID` and `OOD` rows.

The resulting selections:

| cohort         | RI rows                                        | n       | APD rows        |
|----------------|------------------------------------------------|---------|-----------------|
| camelyon       | `camelyon.csv` where `subset == "ID"`          |  20,400 | all 22,402      |
| tcga-4x4       | `tcga_4x4.csv` where `subset == "ID"`          |   5,760 | all 8,160       |
| tcga-2x2       | `tcga_2x2.csv` in full (no OOD rows exist)     | 112,800 | *(APD unused)*  |
| tolkach-esca   | `tolkach_esca_reduced.csv` in full             |   9,000 | all 16,300      |

Tolkach is why this cannot be one column. `tolkach_esca_reduced.csv` is a **balanced
sample**, not a predicate: 500 tiles per (biological class x centre) over UKK / WNS /
CHA_FULL, where WNS and CHA_FULL have 900 available per cell. It cannot be reconstructed
by filtering — only stored. And 3,000 of its 9,000 rows are `VALSET1_UKK`, which APD
classifies as **OOD**. The RI set straddles the APD split.

`VALSET3_TCGA` appears only in the full Tolkach file and is never used by PathoROB's RI —
only by APD, as an OOD cohort.

### The defect this surfaced

Our `pathorob-tcga-4x4` benchmark evaluated all **8,160** rows: the 5,760 `ID` rows plus
2,400 `OOD` rows drawn from four centres PathoROB excludes entirely (Cureline, Greater
Poland Cancer Center, International Genomics Consortium, Johns Hopkins). We were
computing RI over **8 confounder classes where PathoROB has 4**, while describing the
benchmark as faithful. Restricted to `id_ood == "ID"`, our rows are identical to
PathoROB's — so nothing was wrong with the data, only with the selection.

## Decision

Per cohort, one tileset and two manifests:

```
data/pathorob/manifests/
  pathorob-<cohort>.csv       # every tile, carries `apd_split` (ID/OOD). Tileset source
                              # AND the APD view -- APD evaluates ID+OOD, i.e. everything,
                              # so a separate -apd.csv would be a byte-identical copy.
  pathorob-<cohort>-ri.csv    # exactly the rows PathoROB feeds to RI. Carries no
                              # apd_split: for Tolkach the column would actively mislead.
```

Generated reproducibly by `scripts/prep/build_pathorob_views.py` from PathoROB's metadata
CSVs. `tcga-2x2` gets an `-ri.csv` only: APD never evaluates it, and its `subset` column
names the 2x2 pair, not an ID/OOD split.

The four benchmarks are named for their cohorts — `pathorob-camelyon`,
`pathorob-tolkach-esca`, `pathorob-tcga-2x2`, `pathorob-tcga-4x4` — and each reads its
`-ri.csv`. There is no `-full` benchmark: analyses needing every tile (e.g. the
pretraining-overlap study, which contrasts `VALSET3_TCGA` against the other three Tolkach
cohorts) read the **tileset**, as the APD scripts already do.

Consequently:

- `id_ood` is renamed **`apd_split`**. It is APD's notion of in/out-of-distribution and
  nothing else. Naming it generically is what let `tcga-4x4` drift.
- `*-faithful.csv` becomes `*-ri.csv`: "faithful" described our intent, "ri" describes
  what the rows are for. For Tolkach that file is irreplaceable — it *is* the sample.
- `k_max` becomes a uniform `DEFAULT_K_MAX = 100`. The smallest evaluated unit in the
  registry is 1,000 samples, so the ceiling fits inside every subset.

## Consequences

- **Every `tcga-4x4` number changes** — RI, MaRI, CRoMa, k*. The confounder task halves
  from 8 classes to 4. No re-embedding is needed: the tileset holds all 8,160 rows and
  the RI view maps into it.
- `k_max` 25 -> 100 shifts k* and median-k for `prostate*` and `panda*`.
- The four `-ri.csv` views were verified to reproduce PathoROB's row sets **in exact
  order**, and `pathorob-camelyon-ri.csv` / `pathorob-tolkach-esca-ri.csv` are identical
  to the `-faithful.csv` files they replace.
- Anything reading `id_ood` must read `apd_split`.
- The APD study reads the tileset positionally; row order of the all-tiles manifest and
  the tileset manifest was verified to agree for all three APD cohorts. That remains an
  unchecked invariant of the embeddings tree (see ADR-0007).
