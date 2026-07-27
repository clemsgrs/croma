# nAPD averages replicates before taking the ratio

`napd` reduces an `(n_splits, n_iterations)` accuracy matrix by averaging the
replicate axis **first** and taking the skill ratio of the averages
(ratio-of-means). `apd` — PathoROB's reduction, which `croma.downstream` will ship
vendored verbatim per ADR-0011 — keeps the opposite order: ratio per replicate,
then average (mean-of-ratios). This is the single deliberate difference between the
two reductions, and it concerns only the iteration axis.

> The issue that requested this record asked for it as ADR-0012. That number was
> already taken by *paper tooling stays local*, so it lands as 0014.

## Why the orders differ at all

Per replicate the two metrics are related exactly:

```
nAPD_i = APD_i * acc_0i / (acc_0i - chance)
```

The extra factor blows up as a replicate's baseline approaches chance. APD's
denominator is an accuracy, floored at chance by construction and typically 0.5–0.9,
so mean-of-ratios is well-behaved for it. nAPD's denominator is a *skill* — the same
accuracy minus chance — which is free to approach zero. One unlucky replicate is then
enough to dominate the mean of twenty, and it does so with a ratio that carries no
information about the confounder, only about how close that replicate landed to chance.

Observed on real cells, both OOD, both binary (`chance = 0.5`):

| Cell | Mean baseline | Weakest replicate's baseline | Mean-of-ratios | Ratio-of-means |
|---|---|---|---|---|
| prostate / Prost40M | 0.773 | 0.520 (skill 0.020) | **+0.225** | −0.140 |
| pcabiop / MOOZY | 0.790 | 0.580 (skill 0.080) | +0.373 | +0.010 |

Prost40M is the sign flip: mean-of-ratios reports that the confounder *helped*, off the
back of one replicate out of twenty whose baseline landed 0.02 above chance. MOOZY is the
same mechanism short of a sign change — both reductions come out positive there, but
mean-of-ratios inflates the value by a factor of 38. Neither mean baseline is anywhere
near chance, which is the point: a gate on the mean cannot see either case, so with no
gate present (see below) the reduction order is the only thing standing between the
metric and a single replicate's denominator.

## Why the deviation is not a thumb on the scale

The order of the two operations is inert wherever the denominator is healthy, measured
across all 166 model × benchmark × domain cells of the paper's sweep:

| Quantity | Median shift | Max shift | Spearman (mean-of-ratios vs. ratio-of-means) |
|---|---|---|---|
| Accuracy (APD's denominator) | 0.0003 | 0.029 | 0.9989 |
| Skill (nAPD's denominator) | 0.0010 | — near chance, unbounded | — |

So ratio-of-means is not a reduction picked because it flatters nAPD: applied to APD's
own quantity it changes essentially nothing and reorders essentially nobody. It differs
from mean-of-ratios only in the near-chance regime — exactly the regime where the naive
form is unreliable rather than merely noisier.

## Rejected: make both reductions ratio-of-means

The tidier option is one reduction order for both metrics, so the pair differs only in
its normalizer. Rejected. `apd` is reported precisely because it is the faithful
PathoROB reference, and that claim rests on running PathoROB's own code; changing its
reduction order would break bit-identity with upstream and leave the paper reporting a
number that is nobody's APD. The inconsistency is the price of the reference being a
reference.

## Consequences

- `napd` and `apd` are not two parameterisations of one function, and when `apd` lands
  they will not share a reduction. They agree away from chance (Spearman >= 0.94 on every
  tile benchmark) and diverge near it, by design.
- `napd` carries **no gate**: no skill floor, no sentinel, no `None`. Ratio-of-means
  removes the failure mode a gate would have been guarding against, and the floor the
  study-layer implementation still carries (`NAPD_NORM_SKILL_FLOOR = 0.15` in
  `scripts/studies/apd/apd_experiment.py`, retired when that driver moves onto the
  library) sits inside a wide insensitive basin — a cut point with nothing to recommend
  it. Deciding a cell is too imprecise to interpret is a reporting decision, made by
  whoever renders the table. See ADR-0011.
- What `napd` does reject is its domain running out, not a judgement about precision: a
  baseline at or below chance has no positive denominator, so the skill ratio does not
  exist and every ratio below chance would silently invert its sign. That is a
  `ValueError`, not an undefined value.
