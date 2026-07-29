# nIPD integrates normalized degradation over Cramér's V

The primary downstream reduction is normalized integrated performance degradation
(nIPD):

\[
g(V) = \frac{\bar a(V)-\bar a(0)}{\bar a(0)-\pi},
\qquad
\operatorname{nIPD} = \int_0^1 g(V)\,dV,
\]

where \(\bar a(V)\) is mean balanced accuracy across repeated training runs at
Cramér's \(V\), and \(\pi\) is chance performance. `nipd` estimates this integral
by the trapezoidal rule at the supplied Cramér's-\(V\) coordinates.

## Why

APD normalizes a performance change by baseline accuracy. That denominator contains
the irreducible chance floor, so it does not represent how much predictive signal a
model has available to lose. The same absolute degradation can therefore look smaller
for a model with little above-chance baseline skill, and APD values are not directly
comparable across tasks with different chance levels.

nIPD instead divides by baseline skill, \(\bar a(0)-\pi\). The value is therefore a
fraction of the model's available above-chance performance. Correcting unequal baseline
headroom is the metric's central purpose.

The second change is integration. The retired nAPD reduction averaged normalized
degradation equally across sampled splits. That statistic targets the empirical
distribution of the chosen grid: adding points to one part of the Cramér's-\(V\) axis
changes its weight even when the underlying degradation curve is unchanged. nIPD targets
the uniform average over \(V\in[0,1]\), which is also the area because the interval has
unit width. Trapezoidal integration weights each segment by its width,
\(V_s-V_{s-1}\), rather than giving every sampled point equal weight.

Trapezoidal integration does not eliminate discretization or interpolation error. It
states the estimand and prevents grid density itself from silently changing the
weighting; accuracy still depends on how well the observed coordinates resolve a curved
degradation trajectory.

## Reduction order

nIPD first averages balanced accuracy over repeated training runs at each \(V\), then
normalizes the resulting curve. This ratio-of-means prevents a repeat whose baseline
lands barely above chance from dominating through a near-zero denominator. It is an
aggregation choice rather than the conceptual contribution, but it is fixed here for
reproducibility.

APD retains PathoROB's mean-of-repeat-specific-ratios reduction. APD exists as a
faithful continuity analysis, so changing its operation order would defeat its purpose.
ADR-0014 records the detailed rationale for the aggregation order in the retired nAPD
design; the same rationale applies to nIPD.

## Domain and interpretation

The Cramér's-\(V\) vector must be finite, one-dimensional, strictly increasing, aligned
with the rows of the accuracy matrix, and span exactly \(0\) to \(1\). The first row is
therefore the balanced baseline.

Mean baseline balanced accuracy must exceed chance. At chance the denominator is zero;
below chance it is negative and would reverse the interpretation of degradation. This is
the mathematical domain of the metric, not a weak-skill exclusion rule: nIPD imposes no
additional threshold.

nIPD is signed:

- negative values indicate net degradation as the training correlation increases;
- zero indicates no net change;
- positive values indicate net improvement.

## API and compatibility

The public function is `nipd(accuracies, cramers_v, chance)`. The previously released
`napd(accuracies, chance)` function is removed rather than retained as an alias: it
computed a different estimand, so silently redirecting it would change the meaning of
existing calls. There are no external users to migrate at this stage, so the clean API
is preferred over a compatibility shim.

APD remains available as `apd(accuracies)` and is reported only as a compact continuity
analysis with PathoROB. nIPD is the primary downstream reduction.
