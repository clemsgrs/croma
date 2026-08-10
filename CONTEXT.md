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

**Independence group** (`group_id`):
The non-independent source a sample came from — a slide, a patient, a specimen, an
acquisition; whichever unit a study declares. Required on every canonical manifest, as a
non-empty string. Candidates sharing the query's `group_id` are dropped before neighbours
are selected, so a model cannot score well by retrieving near-duplicates of the sample it
is already looking at. Which unit to group by is the study's call, not the library's.
_Avoid_: `slide_id` (the pre-rename name; slide is one possible unit, not the contract),
cluster (that word belongs to the bootstrap, which resamples these same groups).

**Support fraction**:
The fraction of anchors on which RI/MaRI are defined (`1 - ri_undefined_frac`).
Always reported alongside RI/MaRI because those metrics are undefined on
SS-dominated anchors and their pooled values are otherwise computed on a silent,
model-dependent subset.

**Confounder probe**:
The balanced accuracy with which a $k$-NN probe recovers the confounder from the frozen
representation (`confounder_knn_bacc`). Not a robustness metric, but the null model every
robustness metric must beat: across the 25 pathology encoders it rank-predicts every
_pooled_ score — RI, MaRI, the CRoMa median, $F(0)$ — at $|\rho|$ of 0.82–0.97. It cannot
replace them, because it saturates (Hibou-B and Hibou-L differ by 0.0003 in probe accuracy
and by 0.35 in CRoMa), has no zero to be signed about, has a chance level that moves with
the number of confounder classes, and — being a scalar — admits no tail.
_Avoid_: "linear probe" (it is $k$-NN), "shortcut score".

**Confounder-biased probe sweep**:
The downstream protocol APD and nIPD reduce (`croma.probe_sweep`): a *logistic* probe is
trained to predict the **biological class** while a schedule walks the training set from
balanced to fully confounded, and is scored on test rows that do not move. The result is
an `(n_splits, n_iterations)` matrix of balanced accuracies, row `0` the balanced baseline.
Note the confounder is what *biases the training composition* here, not what the probe
predicts — the opposite of the confounder probe above, which is $k$-NN and recovers the
confounder itself. The two share a word and nothing else.
_Avoid_: calling it "the confounder probe", "the linear probe" without saying which.

**Pooled score**:
Any single scalar summarising a model's robustness over a whole benchmark: RI, MaRI, the
CRoMa median, $F(0)$. Contrast with the per-sample CRoMa _distribution_, from which the
tail statistics are read. The probe result is a claim about pooled scores as a class.

**Natural-image control**:
`DINOv2-B` (`plotstyle.CONTROL_MODEL`) — pretrained on LVD-142M, never on a whole-slide
image. A _floor_, not a competitor: it is excluded from rankings, from per-column bolding,
and from every cross-model correlation. Its positive CRoMa (+0.077 mean) is an artifact of
weak structure of either kind — it has the lowest biological $k$-NN accuracy of the panel —
which is exactly what makes it useful: it calibrates what a positive margin is worth on a
representation that encodes little. Never read it as "beats 8 pathology FMs".
_Avoid_: baseline (it is not a competitor), ImageNet control (the corpus is LVD-142M).

**Ranked panel**:
The 25 pathology encoders — the full tile panel minus the natural-image control. Every
statistic that *compares* models is computed over the ranked panel: the cross-model
correlations, the support range, and the rank-agreement $\rho$ (`\<Bench>RankedNModels`,
`SupportRange`, `CromaVsRiRho`, …). Whole-panel *descriptive* statistics keep the control,
because they describe the spread rather than rank within it (`NModels` = 26, `CromaSpan`,
`BioBaccRange`, `ConfBaccRange`). The split is not cosmetic: the control's weak biological
structure can flatter its robustness scores and distort cross-model associations, so it
cannot supply evidence about relative pathology-encoder performance.
_Avoid_: "all models", "the panel" (say which one).

### Tail vocabulary

All tail statistics are read off one object: the empirical CDF of a model's
per-sample CRoMa. Report exactly two of them.

**Confounder-dominant fraction** — $F(0)$:
The CDF evaluated at zero: the fraction of _defined_ evaluation units whose
neighbourhood is confounder-dominant, $\mathrm{CRoMa}_i \le 0$. A _prevalence_ — how
often the margin goes the wrong way. Lower is better, so it is a diagnostic and is
never bolded as a "best". The inequality is closed, so an exactly contested unit
counts as confounder-dominant; the denominator is the defined units only (a manifest
sample each under `all`, a subset occurrence each under `paired_2x2`), which is what
`q_alpha` and `ltm_alpha` are read off too, and `undefined_frac` measures the rest.
CRoMa computes it: read `CRoMaResult.f0` / the stored `croma_f0`, never a fresh
`(samples < 0).mean()` over a per-sample artifact — that is how a boundary or a
denominator drifts.
_Avoid_: $P_{<0}$ (the pre-CDF name, and an _open_ boundary: it differs whenever a
unit sits exactly at zero, which a collapsed or exactly contested neighbourhood
produces), $\hat{F}(0)$ (the manuscript does not hat it).

**Lower-tail mean** — LTM$_{10}$:
The mean of the worst $\alpha$ fraction of anchors, $\mathbb{E}[X \mid X \le Q_\alpha]$.
A _severity_ — how bad the margin gets when it does go wrong. Reported at
$\alpha = 10\%$; its insensitivity to that choice is what the alpha-stability
table establishes. Subscript is the percent as a bare integer, `LTM_{10}`.
_Avoid_: `LTM_{10\%}` (the table's stale form), CVaR, expected shortfall.

**$Q_\alpha$** (the $\alpha$-quantile) is _not_ reported. Its sign carries no
information beyond prevalence, since $Q_\alpha > 0 \iff \hat{F}(0) < \alpha$.
Prevalence and severity are independent; a quantile is neither.

### Evaluation units

The vocabulary of the output layout (see ADR-0007).

Entries below that name `scripts/repro/` paths — *generated artifact*, *caption claim*,
*float basis*, *prose claim*, and the manifest referenced under *run* — describe tracked
paper tooling that is excluded from the Python source distribution (ADR-0012). The canonical
model registry is benchmark-owned at `scripts/bench/model_metadata.csv`, so runtime plotting
does not depend on that excluded paper layer (ADR-0005 and ADR-0017).

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

**Evaluation design** (`all` | `paired_2x2`):
What a benchmark's neighbourhood scope is. `all` — the default everywhere, in the
library and on the CLI — scores every supplied manifest row together, as one scope, at
sample level. `paired_2x2` scores only the manifest's explicitly declared 2x2 subsets,
at occurrence level, and has to be asked for. There are no other values.
_Avoid_: `dataset_wide` (the retired name for `all`; it is rejected, not aliased).

**RI view** (`pathorob-<cohort>-ri.csv`):
Exactly the rows PathoROB's Robustness Index evaluates. The selection differs per
cohort and is _not_ a predicate: Tolkach's is a stored balanced sample. This is what
the four `pathorob-*` benchmarks read.
_Avoid_: "faithful manifest" — it described our intent, not what the rows are for.

**APD view** (`pathorob-<cohort>.csv`):
Every tile of a cohort, carrying `apd_split`. APD evaluates ID and OOD rows, i.e. the
whole cohort, so this doubles as the tileset source. APD is a *study*, not a benchmark:
its paper-reproduction driver lives in `scripts/studies/apd/` and writes to
`output/studies/apd/`. The measurement itself — the probe protocol and both reductions —
is library code (`croma.downstream`, ADR-0011); the study only supplies manifests,
model lists and output paths.

**Skill**:
Balanced accuracy above chance. The quantity a confounder can actually destroy: a probe
scoring at chance has no skill, so it has nothing to lose. Chance is `1/n_biological_classes`
for the benchmark (binary cohorts 0.5, TCGA-4×4 0.25, Tolkach-ESCA 1/6), exact because the
scorer is balanced accuracy. *Normalized skill* rescales it to the fraction of achievable
headroom attained, `(acc − chance)/(1 − chance)`, which is class-count invariant.
_Avoid_: "accuracy" for this — the distinction is the whole point of nIPD.

**nIPD** (normalized integrated performance degradation):
The signed area under the chance-normalized downstream degradation curve. If
`\bar a(V)` is mean balanced accuracy across repeated training runs at Cramér's `V`,
`a_0 = \bar a(0)`, and `π` is chance, the curve is
`g(V) = [\bar a(V) − a_0] / (a_0 − π)` and
`nIPD = ∫_0^1 g(V) dV`. The implementation estimates the integral by the trapezoidal
rule at the observed `V` coordinates, so interval width rather than the number of sampled
conditions weights the curve. Negative values denote degradation, zero denotes no net
change and positive values denote improvement. The mean baseline must exceed chance; there
is no additional weak-skill gate. `APD` is retained as the PathoROB-faithful reference.
_Avoid_: calling nIPD an arithmetic average over splits — that makes the result depend on
how densely a benchmark samples particular parts of the `V` axis. Also avoid calling it
"margin-aware" or an analogue of the RI→MaRI step: the denominator corrects unequal
baseline headroom and the integration summarizes that normalized curve; neither is a
neighbour-distance weighting.

**`apd_split`**:
PathoROB's ID/OOD partition — APD's notion and only APD's. Tolkach's RI view
deliberately straddles it (3,000 of its 9,000 rows are `VALSET1_UKK`, which APD calls
OOD), so this column must never be used to derive an RI row set. See ADR-0008.
_Avoid_: `id_ood` (the old name; its generality is what let `tcga-4x4` silently drift).
_Avoid_: run (a run is a benchmark evaluated at one protocol).

**Protocol**:
The k operating point a metrics run was computed at — `k-star` (each model at its own
kNN-optimal k) or `median-k` (the shared median of per-model k*). It scopes the
metrics tree: `output/metrics/<protocol>/<benchmark>/`. The tile panel (26 models) reports
`median-k`; the **slide panel reports `k-star`**, because with four models the shared median
collapses to a tiny k and starves RI/MaRI of support (27% at k=3 versus 37% at per-model
k*). The median is the *lower* median — an order statistic, so the shared k is always some
model's own k* and therefore on the swept grid.
_Avoid_: mode, setting.

**Run**:
A benchmark evaluated at one protocol — the directory `output/metrics/<protocol>/<benchmark>/`,
holding `results/` (pooled `metrics.csv`, `per_sample_metrics.csv`) and `studies/`. The unit a
paper artifact is derived from. Which run backs which artifact is declared once, in
`scripts/repro/paper_manifest.py`; nothing else may spell a run directory out (there is a test).
Note RI and MaRI are protocol-dependent but CRoMa and LTM are **k-free**: their per-sample values
are bit-identical across the two protocols, so only the count-based metrics move with k.

**Study**:
An analysis that *reads* a run and writes beside it, rather than computing the benchmark's
metrics — the bootstrap CIs, the typed-neighbour ranks, the pretraining-overlap contrast, APD.
A study never chooses a protocol: it inherits the run's, because its numbers sit in the paper
next to that run's RI and MaRI. Studies that named their own protocol all said `k-star`, and
were left reading an archived directory when the tile panel moved to `median-k`.
_Avoid_: experiment (that named the script, not the artifact).

**Tile identity**:
`(sample_id, image_path)` — what makes two manifest rows the same tile. Notably
excludes `label`, which a view attaches to a tile rather than owning: one tile is
`tumor` to `prostate` and `gleason-3` to `prostate-4class`.

**Generated artifact**:
A `.tex` file under `paper/sections/` that a generator owns — the results tables, their
captions, `generated_values.tex`, the model tables. It is rebuilt from
`scripts/repro/paper_manifest.py` by `scripts/repro/build_paper.py`, never hand-edited, and
`tests/test_paper_artifacts.py` fails when it drifts (ADR-0010). Run `build_paper.py` after
every benchmark re-run. Prose sections are *not* generated artifacts; where a generated float
needs authorial prose around it, the prose lives in a `templates/*.tmpl` with a placeholder.
_Avoid_: "the table file" (say whether it is generated), hand-editing a caption number.

**Caption claim**:
A caption sentence that asserts something about the data, paired with a predicate the
generator evaluates before emitting it. A claim that stops holding raises `CaptionClaimError`
instead of shipping. This is why the caption says "no *pathology* model reaches two-thirds
coverage" — the control's 68% support falsifies the unqualified form.
_Avoid_: putting an unchecked numeric claim in a caption; move it to prose with a macro.

**Float basis**:
The one module a generated float's *plot* and its *caption* both read — `_cross_benchmark.py`
for `fig:cross-benchmark`, `_apd.py` for `fig:croma-vs-apd` and `tab:apd-correlation`. Which
models were drawn, which carry an exposure dagger, how many fell below zero: computed once,
rendered twice. A script-drawn PDF beside a hand-typed caption does not drift *apart* — it
drifts *together*, each still faithful to a run that no longer exists, so neither can catch
the other. That is how the cross-benchmark caption kept claiming "TCGA produces no
confounder-dominant models" for a month after the TCGA-4×4 run was corrected to PathoROB's
four in-domain centres and `Prost40M` crossed below zero.
_Avoid_: recomputing a ρ, a roster, or a rank inside a plotting script.

**Prose claim**:
A sentence of body text whose truth depends on a run, paired with a predicate in the generator
that emits the numbers it cites. The numbers reach the page as macros (`\ProvenanceBioBacc`);
the *ordering* they were chosen to illustrate ("the highest biological accuracy in the panel",
"its nearest neighbour once the two probe accuracies are differenced") is asserted, and the
generator raises `CaptionClaimError` rather than emitting a macro that would make the
surrounding sentence false. A caption claim guards a float; a prose claim guards a paragraph.
The distinction that motivates it: a macro keeps a *number* current, but a paragraph is an
*argument about an ordering*, and re-running a benchmark can invert the ordering while every
macro in the paragraph updates itself and stays individually correct — which is exactly what
happened to §3.4 when TCGA-4×4 went from eight medical centres to four.
_Avoid_: hand-typing a number, a rank, or a superlative that a `metrics.csv` can answer.

### Documentation site

**Distribution explorer**:
The documentation site's interactive view of the per-sample CRoMa distribution — the
central evidence object of the work. One widget, master–detail: a permanently visible
**overview** (all panel encoders as compact aligned histogram rows, in the result tables'
sort order), a permanently visible **detail** below it showing the selected encoder's full
histogram with a range brush and a count readout, and an optional **comparison** encoder
overlaid in the detail. It is the site's only distribution view: it replaces the static
per-cohort ridgeline figures, and there is no non-interactive fallback.
_Avoid_: "the histogram widget", "the ridgelines" (retired as site artifacts; the term
survives only for the manuscript's figures).

### Model attributes

Per-model properties carried in the single-source model metadata (see ADR-0005) and
used to annotate figures like CRoMa-vs-scale.

**Robustness-targeted fine-tune**:
A public encoder obtained by fine-tuning a base pathology encoder explicitly for
invariance to acquisition factors. It remains in the ranked panel, but its parent
relationship is reported so it is not mistaken for an independent pretraining run.
_Avoid_: robust model, new foundation model.

**Regime**:
The pretraining paradigm of a foundation model, reduced to a two-way split for
figure encoding: **VLFM** (a vision--language model, e.g. CONCH) vs **vision-only**
(everything else — DINOv2, iBOT, SSL, distilled, etc.).
_Avoid_: modality, family (family is the palette-hue grouping, a different axis).

**VLFM** — Vision--Language Foundation Model:
A model pretrained with paired image--text supervision. One value of _regime_.

**Pretraining tiles** — `n_tiles`:
The number of tiles seen in the **vision self-supervised** stage. For a VLFM this
deliberately excludes the image--text alignment stage: CONCH is `16M` tiles (iBOT)
*and then* 1.17M image--caption pairs (CoCa), of which only the former is recorded.
So `n_tiles` is not "pretraining scale" — it is the vision-SSL budget, and for a VLFM
it omits the supervision the model is named for. A scale analysis across mixed regimes
must say so, or record pairs in their own column.
_Avoid_: "pretraining scale", "training data size" (both imply the total).

**Disclosure markers** — `n/a` vs `n/d`:
`n/a` = _not applicable_ (the natural-image control has no WSIs; the quantity does not
exist). `n/d` = _undisclosed_ (the quantity exists; the authors did not publish it).
Different claims, never interchangeable. Note `pd.read_csv` maps the literal string
`n/a` to NaN unless passed `keep_default_na=False, na_values=[]`.
_Avoid_: VLM, multimodal.

**Pretraining scale**:
How much data a model was pretrained on, measured in **#WSIs** (whole-slide images);
`#tiles` is the finer-grained companion count. The x-axis of the scale figure. Some
values are undisclosed by model cards and sourced from PathoROB with a citation
marker.
_Avoid_: model size (that is the parameter count, a separate attribute).

**Checkpoint family and variant**:
Machine-readable model provenance carried as `family`, `parent_model` and
`variant_role`. A robustness fine-tune remains linked to the parent whose pretraining facts it
inherits; a distilled student names its teacher. `training_run` and `shared_corpus` identify
family members produced together from one disclosed corpus. Plot hue, within-family tone and
canonical order are read from this metadata rather than a second model list in plotting code.
_Avoid_: encoding parent/student or fine-tune relationships only in a display label.

**Institutional/source-domain exposure**:
A conservative benchmark marker used when a model's disclosed institutional corpus and a scored
cohort share a source institution. It means possible source-domain overlap only. Unless source
records establish otherwise, exact patient or slide overlap is unknown and the marker does not
establish leakage.
