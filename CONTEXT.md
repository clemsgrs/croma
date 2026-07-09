# croma

Robustness metrics for pathology foundation models: how much a model's
representation is driven by biology versus by non-biological technical variation
(staining, scanning, tissue prep) across centers. The library computes a small
family of neighbourhood-based robustness metrics and ships the benchmarking
pipeline used in the accompanying paper.

## Language

### The library vs. the metric

**croma**:
The library / Python package (`import croma`) and the project brand. Always
all-lowercase, code font. It is named after its flagship metric but is not itself
a metric.
_Avoid_: CROMA (that styling is reserved for nothing — see CRoMa), Croma.

**CRoMa**:
The flagship metric — the **C**ross-confounder **Ro**bustness **Ma**rgin. A signed,
normalized margin in `(-1, 1)` comparing, for each sample, the distance to its
nearest same-confounder wrong-biology impostor against its nearest other-confounder
correct-biology neighbour: `> 0` biology-dominant (robust), `< 0` confounder-dominant
(fragile), `0` an exactly contested boundary. Canonical styling is the mixed case
**CRoMa** (capitals mark the three source words), mirroring MaRI.
_Avoid_: CCMR, Cross-Confounder Margin Ratio, CROMA, Croma, croma (the lowercase
form is the library, not the metric). It is a margin, not a ratio.

### The metric trio

**RI** — Robustness Index:
The original count-based metric: favourable-minus-unfavourable neighbour counts.
Interpretable but blind to neighbourhood margins. Undefined on SS-dominated anchors.

**MaRI** — Margin-aware Robustness Index:
RI weighted by feature distance rather than raw counts. Mixed-case styling is
canonical. Undefined on the same anchors as RI, so always reported with its
support fraction.

**CRoMa** — see above. The flagship. Defined on every sample (no support caveat),
which is why result tables sort by it.

### Core terms

**Confounder**:
The non-biological grouping variable whose influence on the representation we
measure — e.g. medical center, scanner, data provider. Neighbours are typed by
whether they share the anchor's confounder value.
_Avoid_: batch, site (use only when quoting a specific dataset's column), domain.

**Support fraction**:
The fraction of anchors on which RI/MaRI are defined (`1 - ri_undefined_frac`).
Always reported alongside RI/MaRI because those metrics are undefined on
SS-dominated anchors and their pooled values are otherwise computed on a silent,
model-dependent subset.

### Evaluation units

The vocabulary of the output layout (see ADR-0007).

**Tileset**:
A physical set of tiles that was pushed through the encoders, holding one embedding
matrix per model under `output/embeddings/<tileset>/`. Its `manifest.csv` is the
row-order contract: row `i` of every `<Model>.npy` describes `manifest.csv` row `i`.
The only expensive artifact in `output/`.
_Avoid_: dataset (ambiguous — it named both the tiles and the evaluation).

**Benchmark**:
An evaluable *view* over a tileset: an eval manifest that selects its rows, plus an
evaluation design. Several benchmarks share one tileset (`prostate`,
`prostate-4class` and `prostate-gradebal` all view `prostate-shift`), so a benchmark
never owns embeddings. Camelyon's tileset is 22,402 tiles; the `pathorob-camelyon`
benchmark is the 20,400-tile RI view of it.

**RI view** (`pathorob-<cohort>-ri.csv`):
Exactly the rows PathoROB's Robustness Index evaluates. The selection differs per
cohort and is _not_ a predicate: Tolkach's is a stored balanced sample. This is what
the four `pathorob-*` benchmarks read.
_Avoid_: "faithful manifest" — it described our intent, not what the rows are for.

**APD view** (`pathorob-<cohort>.csv`):
Every tile of a cohort, carrying `apd_split`. APD evaluates ID and OOD rows, i.e. the
whole cohort, so this doubles as the tileset source. APD is a *study*, not a benchmark:
it lives in `scripts/studies/apd/`, is absent from the `croma` library, and writes to
`output/studies/apd/`.

**`apd_split`**:
PathoROB's ID/OOD partition — APD's notion and only APD's. Tolkach's RI view
deliberately straddles it (3,000 of its 9,000 rows are `VALSET1_UKK`, which APD calls
OOD), so this column must never be used to derive an RI row set. See ADR-0008.
_Avoid_: `id_ood` (the old name; its generality is what let `tcga-4x4` silently drift).
_Avoid_: run (a run is a benchmark evaluated at one protocol).

**Protocol**:
The k operating point a metrics run was computed at — `k-star` (each model at its own
kNN-optimal k) or `median-k` (the shared median of per-model k*). It scopes the
metrics tree: `output/metrics/<protocol>/<benchmark>/`.
_Avoid_: mode, setting.

**Tile identity**:
`(sample_id, image_path)` — what makes two manifest rows the same tile. Notably
excludes `label`, which a view attaches to a tile rather than owning: one tile is
`tumor` to `prostate` and `gleason-3` to `prostate-4class`.

### Model attributes

Per-model properties carried in the single-source model metadata (see ADR-0005) and
used to annotate figures like CRoMa-vs-scale.

**Regime**:
The pretraining paradigm of a foundation model, reduced to a two-way split for
figure encoding: **VLFM** (a vision--language model, e.g. CONCH) vs **vision-only**
(everything else — DINOv2, iBOT, SSL, distilled, etc.).
_Avoid_: modality, family (family is the palette-hue grouping, a different axis).

**VLFM** — Vision--Language Foundation Model:
A model pretrained with paired image--text supervision. One value of _regime_.
_Avoid_: VLM, multimodal.

**Pretraining scale**:
How much data a model was pretrained on, measured in **#WSIs** (whole-slide images);
`#tiles` is the finer-grained companion count. The x-axis of the scale figure. Some
values are undisclosed by model cards and sourced from PathoROB with a citation
marker.
_Avoid_: model size (that is the parameter count, a separate attribute).
