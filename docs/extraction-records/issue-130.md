# Issue #130: PathoROB five-encoder extraction

Date: 2026-08-09  
Device: NVIDIA GeForce RTX 3080 Ti (12,288 MiB)  
Runtime: PyTorch 2.7.1+cu128, Transformers 5.14.1, timm 1.0.28  
Policy: FP32 inference and FP32 `.npy` storage

## Fixed scope

Only Mascaret, Phaet, RudolfV 2, RudolfV 2-B, and RudolfV 2-S were extracted, and
only for `pathorob-camelyon`, `pathorob-tcga-2x2`, `pathorob-tcga-4x4`, and
`pathorob-tolkach-esca`. Prostate-shift, PANDA, and PCaBioP were excluded and
byte-audited before and after the run. Metrics were not recomputed.

Authenticated gated-checkpoint reload and deterministic inference passed for all five
models (`tests/test_waiv_smoke.py` and `tests/test_rudolfv2_smoke.py`: 5 passed in
11.42 s). No credential value was printed.

## Batch preflight

The fixed grids below used real Camelyon tiles on the RTX 3080 Ti. Every point passed
the expected shape, FP32 dtype, and finite-value checks. Peak values are PyTorch CUDA
allocator measurements at the selected size.

| Encoder | Grid | Selected | Peak allocated MiB | Peak reserved MiB | Expected dim |
|---|---:|---:|---:|---:|---:|
| Mascaret | 8, 16, 32 | 32 | 5,166.0 | 5,564.0 | 1,536 |
| Phaet | 16, 32, 64 | 64 | 1,898.3 | 2,128.0 | 1,024 |
| RudolfV 2 | 1, 2, 4, 8, 16, 32 | 32 | 6,425.9 | 7,454.0 | 3,072 |
| RudolfV 2-B | 8, 16, 32 | 32 | 1,370.1 | 1,902.0 | 1,536 |
| RudolfV 2-S | 16, 32, 64 | 64 | 1,142.9 | 1,688.0 | 768 |

Waiv outputs were unit-normalized at every grid point. RudolfV 2-family outputs had
the expected non-unit native pooled norms.

## Frozen manifests and local access mirror

The existing output manifests used the retired `slide_id` spelling. Each was
re-derived from the current source, proved byte-identical to the legacy file before
the single `slide_id` to `group_id` header rename, and atomically replaced. Row order
and every cell value were unchanged.

| Tileset | Rows | Legacy SHA-256 | Canonical SHA-256 | Canonical fingerprint |
|---|---:|---|---|---|
| Camelyon | 22,402 | `3e3059dff6a804f16d27006978c1f32b06a6718ebfd7c03e8221d3772dd67caa` | `b9ff0cc20b760219c1a282c7bac7935d7bd8544d03463e13560549d3d0d81a8c` | `6e8cdbb7d7a44175dae924cec2ad97b816589e274ecb2f7dd7325cd7897b0703` |
| TCGA-2x2 | 59,250 | `582336c9cb851714b4737cf9e3f092ed54cbbf922849c8526b2fc3b4f36496e5` | `0b54e5352b531ec7f9ea1809aed9076049f36b431c305af291d854eef1ba2897` | `d0732ca3cc5f39dd9547bebce80f3b6f6a85a72aff8f5eb782a5df8ca304a5db` |
| TCGA-4x4 | 8,160 | `5a8121f707948ae356592d07289e0ec0d15c96423cb960e28806a61588f516f1` | `a43fa14d3e30ce42a9d47f9bb4e72998a4b3f7bf866432e7c8ab3fa3c645dffe` | `f9162721f9f4d41e2f894626ea11132eb7b85424d8b7c09701ecfbec05a47ab9` |
| Tolkach-ESCA | 16,300 | `3a1a5d32c1f8e63a903fced373a2cb264b85970d580974ab65cc1fe0fdff68bb` | `c33aeb9c1f65f25af13b69ccf4ee302d20e7440798562c1bae0841f28a7663ea` | `9de6ff40035f6e11930930d7e99e90e2463e3656a964d272f873e18ad46b7aaf` |

The archive mount stalled during the first extraction attempt; no matrix or sidecar was
published. Tiles were therefore staged from the same authoritative Hugging Face parquet
sources used by `prepare_pathorob.py`: Camelyon revision `b2e762542abbf85dde3f23ec70a2bf1848dcf5c8`,
TCGA revision `6e1dbd4306ebee9759b32503914523e84bddabd0`, and Tolkach-ESCA revision
`c42219a2c168c5995e44487f5747bdacaf4bc2da`. The path maps passed exact one-to-one
coverage and order checks on `(sample_id, canonical_image_path)`, plus exact label,
confounder, independence-group, and basename equality. First/middle/last rows of every
tileset (12 tiles) were byte-identical and pixel-identical between the archive and mirror.
The frozen canonical manifest—not the access path map—remained the artifact fingerprint.

| Tileset | Path-map SHA-256 | Source-manifest SHA-256 | Prepared-metadata SHA-256 |
|---|---|---|---|
| Camelyon | `9bd34538a4449f801c90d1702d44f61119ffcf65db691df93ed82b6d9a2d8f0b` | `385671e7cc4583cbac3db95436291f90cfa8589ce30740ed27f752f223221c57` | `84e0694302effc98b3e8d8506b1ca7052655fdab0c1e9a4999a8d7a9778c4282` |
| TCGA-2x2 | `e04b1a038dc0e3848fcb988e02b8faabf9d59b3348743b892bb4113c953f5126` | `1cba53bc28d1ad22a3c7dd83c36851ba9480dafe7ea532a8809b442f09716e4c` | `1daf1d0ac55eaeea8a828ddba92525641732c03427f78dd8c4dd196eea5fce02` |
| TCGA-4x4 | `b126a5ba1fbf915ca2123ad4d214a841330b4b9e5f6f9e30782aaed7b008e4d9` | `1cba53bc28d1ad22a3c7dd83c36851ba9480dafe7ea532a8809b442f09716e4c` | `1daf1d0ac55eaeea8a828ddba92525641732c03427f78dd8c4dd196eea5fce02` |
| Tolkach-ESCA | `6075323e9fc203bf4039b272e912a9474c8cd31dd5ad136dadae8a50abb71635` | `cc78f322680646fe36735cbea114e8a01e21bab80310dbaaccaec01b1d643790` | `946d68a16da594a77a84557507216fc5afb42cb0d977fb4861d8262db3a61623` |

## Extraction and validation

All 20 matrix/sidecar pairs published successfully. Times below are successful
end-to-end command runtimes in seconds. The peak is the maximum whole-device memory
reported by the 250 ms `nvidia-smi` monitor; total successful command time was
22,390 s (6.22 h).

| Encoder | Camelyon | TCGA-2x2 | TCGA-4x4 | Tolkach-ESCA | Peak device MiB |
|---|---:|---:|---:|---:|---:|
| Mascaret | 923 | 2,299 | 370 | 684 | 5,856 |
| Phaet | 244 | 563 | 119 | 197 | 2,412 |
| RudolfV 2 | 3,080 | 8,120 | 1,135 | 2,242 | 7,738 |
| RudolfV 2-B | 363 | 856 | 168 | 279 | 2,186 |
| RudolfV 2-S | 170 | 351 | 93 | 134 | 1,974 |

The first archive-backed Phaet/TCGA-4x4 attempt was stopped after 309 s when the
workers were blocked on archive reads and the GPU remained idle. The atomic writer
published no matrix or sidecar. The staged retry completed in 119 s and passed every
validation check.

The final validator loaded each matrix from disk, checked the exact frozen-manifest
row count and fingerprint, expected feature dimension, FP32 dtype, all-finite values,
artifact reusability, and complete sidecar provenance (checkpoint revision,
preprocessing, pooling, batch size, dtype, shape, and manifest). It also SHA-256
hashed every matrix and sidecar. All 20 passed. Mascaret and Phaet norms were within
`1.8e-7` of one. RudolfV 2 native norms ranged from 5.96 to 16.67, RudolfV 2-B from
10.20 to 18.23, and RudolfV 2-S from 11.14 to 17.76; every native distribution was
non-degenerate.

An immediate clean compatible-resume pass returned `skipped` for all 20 artifacts,
with 20 zero exit codes and no checkpoint load. Sizes and mtimes for all 40 target
files were identical before and after resume.

The exclusion audit compared SHA-256 for 53 files under prostate-shift, PANDA, and
PCaBioP before and after extraction; all 53 were identical. The pre-existing PathoROB
inventory was exactly 168 files: 21 matrix/sidecar pairs for each of four tilesets,
with no missing or extra path. The original parallel pre-hash log suffered stdout
record interleaving and is therefore not presented as a 168-file hash proof: 127
unique path/hash records were recoverable and all 127 matched. A safely serialized
post-run record contains all 168 SHA-256 values. For the complete 168-file inventory,
the latest mtime and ctime were both Unix time `1783586648`, more than 30 days before
the pre-extraction audit file birth time `1786236119`; no pre-existing matrix or
sidecar was written during this issue.

## Verification

- Pre-overlay focused extraction/artifact tests: 55 passed.
- Final focused extraction/artifact/registry tests: 60 passed in 1.21 s.
- Final full suite: 687 passed, 6 skipped in 35.01 s.
- Final real gated-weight smokes: 5 passed in 12.55 s.
- Artifact validator: 20 passed, 0 failed.
- Compatible resume: 20 skipped, 0 failed; 40/40 sizes and mtimes unchanged.
- Excluded byte audit: 53/53 SHA-256 values unchanged.
- Pre-existing PathoROB audit: 168/168 paths present, 127/127 recoverable pre-hashes
  unchanged, and 168/168 mtimes/ctimes before the extraction boundary.
- Two-axis review against `main`: three Standards findings and three Spec findings;
  all six were addressed (invalid-map tests, shared executable/provenance config,
  lazy access-map resolution on extraction only, staging digests, and final closeout
  evidence). No finding remains unfixed.
