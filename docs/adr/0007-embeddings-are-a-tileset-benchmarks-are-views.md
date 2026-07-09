# Embeddings belong to a tileset; a benchmark is a row-view over one

`output/` has exactly three roots, split along the precious/cheap line:

```
output/embeddings/<tileset>/{manifest.csv, <Model>.npy, <Model>.npy.json}
output/metrics/<protocol>/<benchmark>/{results, plots, studies}
output/studies/<name>/
```

A **tileset** is a physical set of tiles pushed through the encoders. `manifest.csv` is
its row-order contract: row `i` of every `<Model>.npy` describes `manifest.csv` row `i`.
It is written once, by `extract_embeddings.py`, and is the only thing in `output/` that
is expensive to reproduce.

A **benchmark** is a `(tileset, eval manifest, evaluation design)` triple. Its manifest
selects rows from the tileset — and, for `paired_2x2`, may repeat a tile across subsets.
A benchmark therefore never owns embeddings; it gathers a row-view of its tileset's.

A **protocol** is the operating point (`k-star` or `median-k`). It scopes the metrics
tree, so two protocols over the same benchmark sit side by side and diff cleanly.

## Why

Embeddings do not depend on any downstream choice, yet the old layout keyed them by
`(dataset, operating point)`: `benchmark.py --output-dir output/<X>` wrote
`embeddings/ + results/ + plots/` together, so each new operating point re-materialised
the matrices. Camelyon-faithful's embeddings existed in six byte-identical copies
(`-reduced`, `-reduced-kstar`, `-reduced-median`, `-reduced-pruned`, `-reduced-save`,
`faithful/k-star/...`), across five competing names for the same dataset.

The copies also drifted: the full Camelyon tileset carried 21 models while every
faithful copy carried 16, so five newly embedded encoders never reached the benchmarks
that needed them. Under the view model they reach every benchmark over that tileset for
free, because the model roster is *discovered* from what is embedded rather than pasted
into a shell variable.

Two facts made the collapse safe, and were verified against the data before anything was
deleted: every derived embedding set was a **row-subset** of its tileset, and for the
canonical runs the rows were **bitwise identical**.

## Tile identity

The lookup from an eval row to an embedding row is keyed on `(sample_id, image_path)` —
the tile — and on nothing else. In particular **`label` is not part of identity**: the
same tile is `tumor` in `prostate` and `gleason-3` in `prostate-4class`, and the same
slide is cancer `1` in `panda` and ISUP `2` in `panda-isup`. Keying on the full attribute
tuple (the old `build_manifest_row_to_embedding_index`) silently failed for exactly those
two benchmarks. `build_view_row_index` encodes the corrected rule and rejects a view whose
known `sample_id` points at pixels the tileset never embedded.

## Consequences

- `benchmark.py` is pure-read: it never extracts and has no model registry. A model is
  evaluated iff its `.npy` exists in the tileset. Extraction lives solely in
  `extract_embeddings.py --tileset <name>`, which owns `manifest.csv`.
- Benchmarks are addressed by name, not by path: `--benchmark camelyon --protocol median-k`.
  The registry (`scripts/bench/benchmarks.py`) is the single source for a benchmark's
  tileset, manifest, design and `k_max`, replacing the per-driver bash arrays.
- Any script needing a benchmark's embeddings must go through `scripts/bench/views.py`.
  Reading a tileset's `manifest.csv` directly evaluates the **superset**: Camelyon's
  tileset is 22,402 rows while the `camelyon` benchmark is the 20,400-row faithful view.
- Eval manifests are inputs and live under `data/`; `output/` is fully derived except for
  `output/embeddings/`.
- This supersedes the `output/faithful/{k-star,median}/<dataset>/` arrangement, in which
  the `median` tree symlinked embeddings while the `k-star` tree copied them.
