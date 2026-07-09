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
