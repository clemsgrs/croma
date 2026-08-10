# Expanded PathoROB downstream panel (issue #132)

This record covers the downstream computation only. It does not add TCGA-2x2, which has
no APD view, and it does not extend prostate-shift or PCaBiop. The successful private execution
took approximately 28 minutes with eight single-threaded workers. An earlier staging
attempt inherited the sealed baseline's read-only mode and failed before creating a raw
or summary artifact; canonical output was not touched.

## Frozen protocol

- Views: Camelyon, TCGA-4x4, and Tolkach-ESCA.
- Repeats: 20; probe seed: 1000; replicate seeds: `7028, 1624, 6448, 5783,
  1031, 7666, 2715, 8760, 7167, 2135, 3634, 3943, 5967, 8009, 3326, 6004,
  3730, 7504, 2999, 667`.
- Cramér's V grids: Camelyon `0, 1/7, ..., 1`; TCGA-4x4 `0, 0.204124,
  0.353553, 0.5, 0.677003, 0.841625, 1`; Tolkach `0, 1/3, 2/3, 1`.
- Schedule SHA256: Camelyon `606db33a1ef93966816262a282c5a02ab4d86157e4a6def104602a36bbd6df75`;
  TCGA-4x4 `090e1f123b644614b8f18d082aef6bfa35aeaac361ac15af715c21cd4ad69793`;
  Tolkach `db0d9648e884a5be748e7b6fa58d0effb13d5a6d193de9901016079ec155c7d5`.
- Chance levels: `1/2`, `1/4`, and `1/6`, respectively. The nIPD reduction and
  normalization order are unchanged.
- ID/OOD rows: Camelyon 20,400/2,002; TCGA-4x4 5,760/2,400; Tolkach
  10,800/5,500. All 68/192/108 ID pseudo-slide chunks, respectively, contain one
  independence-unit identifier. Tolkach retains its fixed feasible case arrangement.
- Runtime: Python 3.11.15, NumPy 1.26.4, pandas 3.0.3, SciPy 1.17.1, and
  scikit-learn 1.7.2. BLAS thread counts were fixed at one per worker.

The protocol implementation hashes used by the successful execution were:

| Source | SHA256 |
| --- | --- |
| `croma.downstream.probe` | `3ff1ff32e83751cb63f5ce8f988b421d60c8193043b90540966e9027a365d5d0` |
| vendored PathoROB helpers | `9b287d6f45c7efceb0295898a161e2647b77ff776bc6c3f768ee0dc5a8095762` |
| APD reduction | `977435ae583ba22b05ea57e9fc6055311e19a7a4250bf95d67d156b5064726d5` |
| nIPD reduction | `fa1b6659cba626438e553ae8356ac0b66e15ef87d26859a4a5a7a22b9d3a7578` |

## Five additions

Values below are percentages. Full 20-repeat, per-strength accuracy matrices remain in
`output/studies/apd/<view>/<model>.json`; the complete summary is
`output/studies/apd/apd.csv`.

| View | Encoder | nIPD ID | nIPD OOD | APD ID | APD OOD |
| --- | --- | ---: | ---: | ---: | ---: |
| Camelyon | Mascaret | -0.74 | -0.28 | -0.58 | -0.27 |
| Camelyon | Phaet | -6.42 | -2.02 | -4.42 | -1.69 |
| Camelyon | RudolfV 2 | -1.99 | 0.12 | -1.40 | 0.05 |
| Camelyon | RudolfV 2-B | -1.95 | -0.03 | -1.41 | -0.02 |
| Camelyon | RudolfV 2-S | -0.77 | 0.05 | -0.64 | -0.02 |
| TCGA-4x4 | Mascaret | -0.40 | 0.98 | -0.37 | 0.90 |
| TCGA-4x4 | Phaet | -1.37 | -2.88 | -1.13 | -2.28 |
| TCGA-4x4 | RudolfV 2 | -0.39 | 0.92 | -0.25 | 0.65 |
| TCGA-4x4 | RudolfV 2-B | -0.45 | 1.07 | -0.32 | 0.80 |
| TCGA-4x4 | RudolfV 2-S | -0.27 | 0.91 | -0.21 | 0.68 |
| Tolkach | Mascaret | 0.58 | 0.09 | 0.65 | 0.10 |
| Tolkach | Phaet | -0.26 | -0.03 | -0.43 | -0.10 |
| Tolkach | RudolfV 2 | 0.02 | -0.04 | 0.09 | -0.00 |
| Tolkach | RudolfV 2-B | 0.04 | -0.00 | 0.09 | 0.00 |
| Tolkach | RudolfV 2-S | 0.56 | 0.17 | 0.65 | 0.20 |

The summary explicitly records these relationships:

- RudolfV 2 is the teacher; RudolfV 2-B and RudolfV 2-S are distilled students.
  Their downstream susceptibility stays close to the teacher. The largest absolute
  student-teacher nIPD movement is 1.22 percentage points (RudolfV 2-S, Camelyon ID).
- Mascaret is the acquisition-robustness fine-tune of Midnight-12k. Its nIPD magnitude
  moves toward zero on Camelyon ID, both TCGA-4x4 regimes, and Tolkach OOD. It moves
  slightly away from zero on Camelyon OOD (0.22% to -0.28%) and Tolkach ID (0.16% to
  0.58%). This is a mostly favorable but non-uniform change.
- Phaet is the acquisition-robustness fine-tune of Phikon-v2. Both ID and OOD nIPD move
  toward zero on every view (improvements of 3.87--13.45 and 1.26--15.10 percentage
  points, respectively).

## Baseline comparison and correlations

The 63 pre-existing PathoROB raw cells are byte-identical. All six numerical fields of
all 83 old `apd.csv` rows are exactly unchanged; the deterministic whole-file rewrite
only adds relationship/control columns and the 15 new rows. DINOv2-B remains present in
each joined view but excluded from ranks and correlations, leaving 25 ranked encoders.

| Endpoint | View | CRoMa rho, 20-model baseline | CRoMa rho, 25-model panel |
| --- | --- | ---: | ---: |
| nIPD ID | Camelyon | 0.883 | 0.938 |
| nIPD ID | TCGA-4x4 | 0.932 | 0.908 |
| nIPD ID | Tolkach | 0.932 | 0.948 |
| nIPD OOD | Camelyon | 0.641 | 0.742 |
| nIPD OOD | TCGA-4x4 | 0.851 | 0.891 |
| nIPD OOD | Tolkach | 0.773 | 0.812 |

Across all 36 tile-view combinations of target (nIPD/APD, ID/OOD) and representation
metric (CRoMa/RI/MaRI), rho movement ranges from -0.024 to +0.104. No sign changes and
no significance-threshold changes occur; every expanded-panel association remains
positive and significant (`rho >= 0.713`, `p <= 6.32e-5`). The original conclusion is
therefore unchanged. The family comparisons above are new evidence made possible by the
five additions, rather than reversals of a baseline conclusion.

## Seals and publication artifacts

- Pre-publication tree: 114 files, tree SHA256
  `1de78cd1d06f014ee52ad4ebcc3687a5e474be48881d85e854c2853394a6142f`.
- Fifteen new raw cells: aggregate SHA256
  `f8867a84a8b395de2de9aba33f407703747f5bee63087f5550bb820108d3fbc6`.
- Summary SHA256: `b2eee5e34a6de7e9da3a9b54d836d8e5f150599e3baf94fad5b00eeb31128103`.
- Joined metrics SHA256: `ca1326ac9c3f7a42c1f4da00aedbff11781ed3815983ec5e6cf8953d499903ba`.
- Correlations SHA256: `64723e756c06ea40a19ef0b608c2c6b11c4b448f5d1c92bc0c2f4dfc6821475b`.
- Complete private downstream tree after canonical plot regeneration: SHA256
  `325c51a66eaa91538cc8d7d80f1c81adc2c0e345eb58d61a915b3d5b4bb15d90`.

The historical PCaBiop `k-star` metrics file is no longer present in the active output
tree. Because PCaBiop is out of scope here, its four exact metric rows were recovered
from the sealed pre-publication join solely to regenerate the combined file. All old
PCaBiop joined values and correlation rows remain exact; only the new relationship/control
columns are added.
