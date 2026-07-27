# Downstream shortcut susceptibility ships in the library

`croma.downstream` ships the confounder-biased probe protocol *and* both of its
reductions (`apd`, `napd`). This narrows ADR-0002: the library is no longer
representation-metrics-only.

## Why

nAPD is a named metric contribution of the paper, and the manuscript tells readers
that metric implementations are released as the `croma` package. A contribution that
ships in no package cannot be reused, and the sentence would be false for it.

The reduction alone was the obvious lean compromise — export `napd(accuracies, chance)`
as a pure function and leave the protocol in `scripts/`. It was rejected because the
accuracy matrix is the hard part to produce: a caller who cannot run the confounder-biased
split sweep cannot obtain the input, so shipping only the reduction ships only the easy
half. Reproducing shortcut susceptibility on a new model is the actual use case.

The cost ADR-0002 was guarding against does not materialise here. The protocol is
numpy/sklearn-only, and `scikit-learn` is already a hard dependency, so
`croma.downstream` adds **zero** new install weight. Torch stays confined to `[repro]`,
where it is needed for embedding extraction — which remains out of the library.

## Shape

- **`croma.downstream`** — the probe protocol plus `apd` and `napd`. Consumes
  embeddings and a split assignment; never loads a model or reads a manifest.
- **`croma/downstream/_pathorob.py`** — `compute_apd`, `get_patches_map_to_split` and
  `train_logistic_regression`, vendored **verbatim** from PathoROB (BSD 3-Clause,
  © 2025 BIFOLD Pathomics), licence notice retained and credited in `NOTICE`.
  Verbatim rather than reimplemented, because the paper reports APD as the faithful
  PathoROB reference and that claim requires bit-identical code. Vendored rather than
  depended on, because PathoROB is not on PyPI and PyPI rejects direct-URL
  dependencies, which would block the croma release.
- **`scripts/studies/apd/`** — retained, reduced to a driver: manifests, model lists,
  output paths, CSV assembly. No metric logic.

## Consequences

- ADR-0002's "the contribution is the metrics (RI, MaRI, CRoMa)" now reads
  "(RI, MaRI, CRoMa, nAPD)". The `scripts/`-is-not-a-product principle is unchanged.
- Vendored code is a sync liability: upstream PathoROB fixes do not propagate. Accepted
  — the vendored functions are the frozen definition of the reference metric, so
  *not* tracking upstream is the intended behaviour.
- nAPD carries no gate or skill threshold. It is an estimator; deciding a cell is
  unmeasurable is a reporting concern, not part of the metric.
