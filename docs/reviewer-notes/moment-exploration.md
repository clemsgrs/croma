# Moment exploration: do distributional moments add value beyond median + LTM?

Decision-support study (cache-only; no benchmark / embedding / neighbour recompute).
Reproduce with `python scripts/experiments/moment_exploration.py`.
All numbers below come from `output/moment_exploration/{moment_summaries.csv,moment_analysis.json}`.

- Per-sample CRoMa read from each benchmark's `results/per_sample_metrics.csv`, column
  `croma_m5` (headline radius m=5; `croma_m1` used as a sensitivity). Non-finite dropped.
- LTM_alpha / Q_alpha reuse `croma.metrics.tail.compute_tail_metrics` (identical to the paper).
- Conventions: `std`/`var` population (ddof=0); `skew` Fisher-Pearson (scipy bias=True);
  `kurt` Fisher **excess** (normal = 0). Lower partial moments use the fragile set
  {CRoMa_i < 0}: `downside_semivar_0 = mean(x^2 | x<0)`, `downside_semidev_0 = sqrt(.)`,
  `downside_third_0 = mean(x^3 | x<0)`, `downside_skew_sub = skew(x[x<0])`,
  `downside_semivar_med = mean((x-median)^2 | x<median)`, `frac_below_0 = P(x<0)`.
- Benchmarks: camelyon, tcga2x2, tcga4x4, tolkach (16 models each); prostate (16);
  panda-isup (4, low-power — flagged, not dropped).

## 0. Sanity vs cache (max |recomputed - cached|, m=5)

| benchmark | median vs `croma` | LTM10 vs cache | Q10 vs cache |
|---|---|---|---|
| camelyon (all) | 1e-16 | 8e-17 | 8e-17 |
| tcga4x4 (all) | 6e-17 | 7e-17 | 7e-17 |
| tolkach (all) | 6e-17 | 1e-16 | 9e-17 |
| prostate (all) | 1e-16 | 1e-16 | 8e-17 |
| tcga2x2 (paired 2x2) | 0.038 | 6e-17 | 8e-17 |
| panda (paired 2x2) | 0.011 | 0.0 | 8e-17 |

LTM10/Q10 match the cache to machine precision **everywhere** (they are pooled per-sample
statistics). The pooled median matches `croma` to machine precision for the four all-rows
benchmarks; for the two paired designs it differs by 0.01-0.04 **by construction** — the cached
`croma` there is the median of per-occurrence medians, not the pooled sample median. Recomputation
is faithful.

## 1. Raw symmetric moments are the wrong tool for one-sided risk (confirmed)

Mean |Spearman vs LTM10| across the 5 strong (16-model) suites, m=5:

| summary | mean \|rho\| vs LTM10 | note |
|---|---|---|
| **std / var** | **0.274** | weakest of all; sign flips (-0.62 prostate … +0.80 panda) |
| kurt | 0.459 | sign-unstable (+0.76 prostate, -1.00 panda) |
| skew | 0.510 | sign-unstable (+0.71 tcga2x2, -0.63 tolkach) |
| frac_below_0 (count) | 0.707 | consistent sign, moderate |
| downside_semidev_0 | 0.659 | directional |
| downside_skew_sub | 0.656 | directional but brittle |
| downside_third_0 | 0.675 | directional |
| **downside_semivar_0** | **0.785** | strongest downside LPM |
| Q20 | 0.783 | tail family |
| LTM20 | 0.938 | tail family |
| LTM05 | 0.952 | tail family |
| Q05 | 0.974 | tail family |

Three clean tiers emerge: **tail family (~0.91 avg) > downside LPMs (~0.66-0.79) > raw symmetric
moments (~0.27-0.51)**. Variance/kurtosis are nearly uncorrelated with the fragile tail and even
flip sign across benchmarks — symmetric spread is not measuring the one-sided robustness risk.
This **confirms the a-priori view**: the principled analog is directional (downside), not symmetric.
`mean` is redundant with `median` (Spearman +0.96 to +1.00 vs median) and adds nothing.

Sensitivity at m=1 is essentially identical (downside_semivar_0 0.769, std/var 0.309, skew 0.521,
kurt 0.464) — no conclusion depends on the averaging radius.

## 2. But the downside LPMs do NOT beat LTM's own family — they are its noisier cousins

The correct question is not "downside vs raw" (downside wins) but "downside LPM vs the LTM/Q
family." Here the LPMs **lose**:

- No downside LPM reaches the alignment that LTM10's own alpha-siblings do: best LPM
  `downside_semivar_0` = 0.785, vs **LTM05 0.952, LTM20 0.938, Q05 0.974**.
- The LPMs are **sign-unstable across benchmarks**, which a headline summary must not be:
  `downside_semivar_0` vs LTM10 is -0.81/-0.84/-0.82/-1.00/-0.97 on five suites but flips to
  **+0.48 on tolkach**; `downside_skew_sub` is -0.94/-0.70/-0.79/-1.00/-0.59 but **+0.27 on tolkach**.
  The flips occur exactly where the fragile set {x<0} is small (tolkach medians are strongly
  positive), so the LPM is estimated on a sparse, idiosyncratic tail.
- `downside_semivar_med` (median-target variant) is the weakest LPM (0.382) and is actively
  misleading: it is self-referential, so a uniformly bad model with a compressed spread *below its
  own very-negative median* looks "low-risk" (see the Hibou-L exception below).

So the downside LPMs carry roughly the same directional information LTM already encodes, but with
lower fidelity and unstable sign. They are **redundant-in-direction with LTM, not orthogonal**.

## 3. Alpha-stability defends alpha = 0.10

Spearman of model rankings across alpha (16-model suites):

| | 0.05 vs 0.10 | 0.10 vs 0.20 | 0.05 vs 0.20 |
|---|---|---|---|
| LTM (min across suites) | 0.90 | 0.88 | 0.69 |
| LTM (max) | 0.98 | 0.99 | 0.94 |
| Q (min) | 0.87 | 0.90 | 0.70 |

LTM rankings barely move between alpha = 0.05, 0.10, 0.20 (adjacent-alpha Spearman >= 0.88 on every
suite), and LTM is at least as stable as the raw quantile Q. alpha = 0.10 is a robust middle choice,
and averaging the tail (LTM) rather than reading a single quantile (Q) is the more stable estimator.

## 4. Exceptions (largest rank gap vs LTM10) — the count fails, the LPMs are brittle

| benchmark | summary | model | LTM rank -> summary rank | reading |
|---|---|---|---|---|
| **prostate** | **frac_below_0 (count)** | **Midnight-12k** | **14 -> 1** | count says best, severity says near-worst |
| tcga4x4 | kurt | CONCHv1.5 | 2 -> 16 | raw symmetric moment reorders spuriously |
| tolkach | downside_skew_sub | Prost40M | 15 -> 1 | skew on a sparse {x<0} set = noise |
| tcga2x2 | downside_semidev_0 | Virchow | 14 -> 2 | LPM conflates rare-but-severe with mild |
| camelyon | downside_semivar_med | Hibou-L | 16 -> 5 | self-referential median target masks the worst model |
| panda | downside_semivar_med | MOOZY | 1 -> 2 | (4-model suite; negligible) |

The one **material, interpretable** exception is `frac_below_0` on **Midnight-12k (prostate)** — and it
is a **count-vs-severity failure that supports the existing design**:

- Midnight-12k has the **fewest** confounder-dominant samples in the suite (frac_below_0 = 0.499,
  rank 1) but a **catastrophic** lower tail (p10 = -0.41, p5 = -0.54, p1 = -0.76, min = -0.88),
  giving LTM10 = -0.566 (rank 14).
- H-optimus-0 has **more** fragile samples (frac_below_0 = 0.63) but only **mild** ones
  (p10 = -0.21, p1 = -0.46), so LTM10 = -0.311 ranks it #1.
- Both severity-aware measures agree it is fragile: LTM10 rank 14 and `downside_semivar_0` rank 12
  (they correlate at rho = -0.97 in prostate). Only the **count** (frac_below_0, rho = -0.36 vs LTM)
  gets it wrong.

This is the paper's RI-vs-CRoMa "beyond counts" thesis reproduced at the summary level: the count
is blind to severity; LTM (and, equivalently here, downside semivariance) is not. It argues **for**
the severity-aware LTM headline and **against** adding a count — not for adding a new LPM.

## Recommendation

**Keep median + LTM10 only. Do not add downside semivariance or downside skewness as companions.**

1. Raw symmetric moments (variance, kurtosis) are inferior for the one-sided question — mean |rho|
   vs LTM10 = 0.27 / 0.46 with unstable sign. Confirmed, but this only argues for a *directional*
   downside summary, which LTM already is.
2. The downside lower partial moments are **redundant-in-direction with LTM** (they agree with it on
   the one case that matters, Midnight-12k: semivar rank 12 ~ LTM rank 14) yet **less stable** than
   LTM's own alpha-family (best LPM 0.785 vs LTM05 0.952 / LTM20 0.938 / Q05 0.974) and **sign-flip**
   on sparse-tail benchmarks (tolkach). They add noise, not orthogonal signal.
3. Their disagreements with LTM (rank gaps up to 14) trace to known pathologies — sparse {x<0} sets
   (downside skew), self-referential targets (median-threshold semivariance), frequency/magnitude
   conflation (semideviation) — i.e. brittleness, not insight.
4. alpha = 0.10 and the tail-mean (vs single quantile) are empirically defended (Spearman >= 0.88
   adjacent-alpha; LTM >= Q in stability).

**Optional supplementary (not a metric change):** report `frac_below_0` (confounder-dominant
fraction) **only** as the count foil whose Midnight-12k failure motivates a severity-aware headline —
it strengthens the "beyond counts" narrative. Do **not** promote downside semivariance/skewness:
where they are right they merely echo LTM, and where they diverge they are wrong.
