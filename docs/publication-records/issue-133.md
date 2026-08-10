# Expanded five-encoder publication record (issue #133)

## Published scope

The coherent public panel is **25 ranked pathology encoders plus DINOv2-B**, the unranked
natural-image control, on Camelyon, TCGA-4x4, and Tolkach-ESCA. **TCGA-2x2 remains supplementary/local**.
Prostate-shift, PANDA, and PCaBiop were not expanded.

The committed web artifacts are `results/{camelyon,tcga-4x4,tolkach-esca}.csv`,
`results/cross_benchmark.csv`, `results/distributions.json`, and
`results/PROVENANCE.json`. The complete local runs are under
`output/metrics/median-k/pathorob-*/`; embeddings and completion sidecars are under
`output/embeddings/pathorob-*/`; downstream raw cells and reductions are under
`output/studies/apd/`. The manuscript is a generated, git-ignored build tree under
`paper/`.

## Replay recipes

Checkpoint environments and authenticated smoke commands are pinned in
`docs/benchmarking.rst`. Extraction is one model and tileset at a time so each selected
batch is explicit:

```bash
PYTHONPATH=src:scripts/bench python scripts/bench/extract_embeddings.py \
  --tileset <tileset> --manifest <source-manifest> \
  --models <model> --batch-size <selected-batch> \
  --image-path-map <validated-local-access-map> --device cuda
```

The selected batches were Mascaret 32, Phaet 64, RudolfV 2 32, RudolfV 2-B 32, and
RudolfV 2-S 64. Exact checkpoint revisions, preprocessing, pooling, source revisions,
manifest fingerprints, mirror digests, dimensions and per-cell runtimes are in
`docs/extraction-records/issue-130.md` and each `.npy.json` completion sidecar. The 20
successful extraction commands took **22,390 s** (6.22 h) in total on an RTX 3080 Ti.

Each robustness run is replayed from its `run_config.json` with forced computation:

```bash
PYTHONPATH=src:scripts/bench python scripts/bench/benchmark.py \
  --benchmark <pathorob-benchmark> --protocol median-k \
  --k-grid sparse --recompute-metrics
PYTHONPATH=src:scripts/bench python scripts/bench/render.py \
  output/metrics/median-k/<pathorob-benchmark>
PYTHONPATH=src:scripts/bench python scripts/tools/export_results.py
PYTHONPATH=src:scripts/bench python scripts/tools/build_doc_figures.py
PYTHONPATH=src python scripts/repro/build_paper.py
```

The downstream additions use the unchanged 20-repeat protocol recorded in
`docs/downstream-records/issue-132.md`. Eight single-threaded workers completed the private
execution in approximately 28 minutes.

## Operating points and movement

All four complete robustness runs use automatic per-model MaRI temperature, CRoMa
`m = 5`, `alpha = 0.10`, and the sparse PathoROB `k` grid. The shared lower-median
operating points did not move when the five encoders joined:

| Cohort | shared k |
| --- | ---: |
| Camelyon | 11 |
| TCGA-2x2 | 61 |
| TCGA-4x4 | 71 |
| Tolkach-ESCA | 61 |

For the original 21 rows, all 31 comparable numeric fields show **exact zero movement** in
every cohort. Rank numbers for the 20 pathology encoders do move because five competitors
were inserted: old mean-rank
values increase by 3.3335–5.0 positions even though their underlying scores are unchanged.
The aggregate frontier changes from CONCHv1.5, GenBio-PathFM, CONCH, and H-optimus-1 in the
historical panel to Mascaret alone in the expanded panel. Mascaret leads the expanded table
at mean rank 1.6665; RudolfV 2-S, RudolfV 2, RudolfV 2-B, and Phaet enter at 4.333, 4.5,
5.5, and 10.0, respectively. DINOv2-B retains its three cohort measurements but is
unranked and excluded from the frontier.

The supplementary typed-neighbour-depth analysis remains a deliberately fixed historical
panel rather than an expanded result. Its read-only source is the sealed
`output/metrics/_archive/issue-131-prepublish-full-runs-20260810T012511Z/runs/pathorob-camelyon/`
snapshot: the 20-pathology-plus-control summary CSV is
`2facf7719b38ae7890d01dafe5b7b90c61b53a4409bba400c37a525d05fabc26`, its JSON is
`6f6c981bbad86383af05c4cd3741114423650de94149fbda45942e6225d17239`, and its co-located
metrics CSV is `fa87db10dfc767e2b5b217a2f1cea89c95985bbf87853c139e48900040dc2aa6`
(SHA-256). The paper renderer validates all three hashes, the exact roster, the independent
JSON count, and the shared `k = 11` before emitting the table. It never joins this historical
study to the live 25-encoder metrics or correlations.

## Material conclusions

The added families strengthen rather than reverse the central result. Across all 36
downstream combinations of reduction, ID/OOD arm, cohort, and representation metric,
Spearman movement is -0.024 to +0.104. There are **no sign changes** and no
significance-threshold changes; every expanded association remains positive and significant
(`rho >= 0.713`, `p <= 6.32e-5`).

The new within-family comparisons are material. Mascaret's nIPD magnitude moves toward zero
relative to Midnight-12k in four of six cohort/arm cells, with small regressions on Camelyon
OOD and Tolkach ID. Phaet improves over Phikon-v2 in all six. The two RudolfV 2 students stay
close to their teacher; the largest absolute student/teacher nIPD movement is 1.22 percentage
points. RudolfV 2-family results on Tolkach retain the conservative Charité/CHA
institutional/source-domain caveat and are not presented as proof of record leakage.

## Integrity and deferred pooling question

Issue #131 retained content-addressed rollback archives and proved the four previous runs,
14,486 unaffected output entries, 104 embedding fingerprints, and four evaluation manifests
stable. Issue #132 proved all 83 prior downstream summary rows numerically exact and all 63
prior tile raw cells byte-exact. This publication pass reads those artifacts but does not
rewrite embeddings, metric/APD runs, or their sealed archives.

Pooling sensitivity is intentionally separate. Follow-up #124 compares the current RudolfV
2 family contract—concatenated CLS with mean-pooled patch tokens—with an L2-normalized CLS
embedding and asks whether any robustness or downstream conclusion changes.
