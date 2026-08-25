# Pooling-sensitivity publication record (issue #124)

## Published scope and evidence

This matched study compares the canonical representation with one alternative for five
encoders on Camelyon, TCGA-2x2, TCGA-4x4, and Tolkach-ESCA. Canonical Mascaret and Phaet
use checkpoint-native `model.encode` output and normalization; their alternative
concatenates raw CLS and mean-patch tokens without output normalization. Canonical
RudolfV-2, RudolfV-2-B, and RudolfV-2-S concatenate CLS with the mean of 784 patch tokens
after excluding eight register tokens; their alternative is the unnormalized raw CLS
token alone. The canonical representations remain the published default throughout.

The complete local evidence is under `output/studies/pooling-sensitivity/`. The source
tables are `results/comparisons.csv` and `results/rankings.csv`; the 20 deterministic
paired occurrence archives are under `per-occurrence/`; and `report.md` and
`run-provenance.json` record the reproducible #152 bundle. The #153 interpretation is
machine-published under `analysis/`, with `analysis/impact-summary.csv`,
`analysis/recommendations.csv`, `analysis/summary.json`, and `analysis/report.md` as the
shortest entry points. This record copies analytical facts from those files and does not
independently recompute or reclassify them.

## Matched protocol

Both representations use the same manifests and the same fixed k: 11 for Camelyon, 61
for TCGA-2x2, 71 for TCGA-4x4, and 61 for Tolkach-ESCA. MaRI tau is selected separately
per representation at fixed k. Biological k* uses the sparse grid and the smallest-k tie
break. The Camelyon production sweep extends to k = 600; its k = 300 diagnostic cap gives
the same canonical-to-alternative k* movements: Mascaret 7 to 7, Phaet 9 to 11,
RudolfV-2 21 to 21, RudolfV-2-B 11 to 9, and RudolfV-2-S 7 to 11.

Paired uncertainty is the alternative-minus-canonical headline CRoMa contrast, resampled
by shared `group_id` with 2,000 bootstrap replicates, seed 0, and a two-sided 95% interval
from NumPy's linear percentile. The decision rule changes pooling only when at least one
public-cohort CRoMa benefit is supported and there is no CRoMa, LTM10, sign, rank,
frontier, or family regression. It has no minimum effect-size threshold.

The hash-linked alternative sidecars contain the complete preprocessing and pooling
contracts, manifest fingerprints, shapes, and FP32 dtypes. Their checkpoint revisions are
`e95e7ea15e039e78d74def101415e19d9a67ba80` (Mascaret),
`e0ce6e0ee248470bd8604823e412ca64048a2495` (Phaet),
`482d9519c6a10fc22fbe5bcd6a87d5daf056643c` (RudolfV-2),
`b2cb55c8fff8aaaf9cc16fda6d09bfb21dfc6db8` (RudolfV-2-B), and
`76abacd512a98c72a6db6192af9fc98313c3bd78` (RudolfV-2-S).

## Numerical impact and decision

The table is the five-row summary from `analysis/impact-summary.csv`. CRoMa deltas are
alternative minus canonical; an asterisk means that the shared-group 95% interval strictly
excludes zero. Supported counts include all four cohorts, while public LTM10 counts use
Camelyon, TCGA-4x4, and Tolkach-ESCA.

| Encoder | Alternative | CRoMa delta (Cam / 2x2 / 4x4 / Tolkach) | Supported CRoMa (+/-) | Public LTM10 (+/-) | Sign crossings | Rank | Family reversals | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Mascaret | cls-mean-patch | -0.0118* / -0.0259* / -0.0206* / -0.0165* | +0 / -4 | +2 / -1 | 0 | 1 to 1 | 0 | retain |
| Phaet | cls-mean-patch | +0.0058* / +0.0003 / -0.0006 / +0.0088* | +2 / -0 | +0 / -3 | 0 | 10 to 10 | 0 | retain |
| RudolfV-2 | cls-only | -0.0020 / +0.0035* / +0.0041* / -0.0203* | +2 / -1 | +0 / -3 | 0 | 3 to 2 | 3 | retain |
| RudolfV-2-B | cls-only | -0.0031 / +0.0129* / +0.0122* / -0.0091 | +2 / -0 | +0 / -3 | 0 | 4 to 4 | 1 | retain |
| RudolfV-2-S | cls-only | -0.0022 / +0.0102* / +0.0154* / -0.0143 | +2 / -0 | +0 / -3 | 0 | 2 to 3 | 2 | retain |

The machine classification records three qualitative claim changes, all confined to the
Rudolf family reversals enumerated below. It records no CRoMa or LTM10 zero crossings and
no tie transitions. Mascaret remains the sole public frontier member. The internal
five-encoder order remains Mascaret, RudolfV-2-S, RudolfV-2, RudolfV-2-B, Phaet; in the
public panel only the two leading Rudolf variants swap within the top four. Mascaret versus
Midnight-12k and Phaet versus Phikon-v2 retain every tested family relation. The analysis
did not emit a separate broader manuscript-claim boolean, so this record does not infer
one.

The three machine-classified reversals are confined to the Rudolf family: teacher versus
B on Tolkach CRoMa, teacher versus S on Tolkach LTM10, and teacher versus S on combined
rank. Because each alternative has at least one guard regression, the recommendation is
to retain canonical pooling for all five encoders.

## Replay and integrity closeout

The exact normal and `--evaluate-only --check` commands are in `run-provenance.json`.
The derived interpretation is checked with:

```bash
python scripts/studies/pooling_counterfactual.py \
  --study-root output/studies/pooling-sensitivity --check
```

The closeout audit verified all 40 canonical matrices and sidecars against the frozen
SHA-256, size, and mtime baseline. All 20 alternatives remain below the study-owned
`embeddings/` tree, outside canonical `output/embeddings/` discovery. The 23 declared
#152 outputs and 10 declared #153 outputs match their hashes and sizes, and their tables,
recommendations, family relations, ranks, ties, orders, and frontiers are cross-file
consistent. A normal #153 replay returned `reused`; check mode returned `checked`; both
preserved the exact size and mtime of all 76 study files. Force writes are confined to
`analysis/`, including traversal and symlink-ancestor guards.

All 20 NPZ archives have fixed ZIP metadata, alphabetical members, and exact aligned
`occurrence_index`, `source_sample_index`, `subset`, `sample_id`, and `group_id` arrays.
Their source indices reproduce the manifests, and the five encoders have identical
identity arrays within each cohort.

One provenance limitation is explicit: #152 did not persist per-run runtime fields.
Operational session observations were approximately 3 h 26 min for the full 20-cell
publication and 3 h 15 min for its full no-write check, but these are not machine-provenance
values and are not reconstructed from mtimes. The bundle directly proves canonical
embedding preservation and tracked public-result hashes; broader ignored metric and
publication trees were protected by target confinement rather than a dedicated pre-run
hash/mtime baseline.
