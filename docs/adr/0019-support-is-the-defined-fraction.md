# Support is the defined fraction

RI and MaRI expose one shared `support`: the fraction of evaluation units on
which both metrics are defined, because distance weighting changes scores but
not which units contribute. It is required when a `RobustnessResult` is
constructed. The result also carries the three shared cause fractions
`ss_dominated_undefined_frac`, `oo_dominated_undefined_frac`, and
`mixed_undefined_frac`, whose denominator is every requested unit.

The benchmark persists exactly the same shared schema. RI and MaRI must agree
on all four fields before a row is written, so metric-prefixed copies would
duplicate one fact and could only drift. There is no complementary aggregate:
callers that need the unsupported share can derive it from `support`, while the
named causes retain the diagnostic detail.

## Why this is a clean break

The positive field states the quantity readers need beside RI and MaRI. Keeping
its complement during a staged migration created two authorities for the same
denominator, and defaulting support to full silently made hand-built results
look more complete than the evidence justified. The required constructor
argument makes omission fail immediately.

Cached summaries are accepted only when their keys match the current schema
exactly. A cache containing a retired field, or missing `support` or a shared
cause, is stale and the metric is recomputed. It is not converted: conversion
would certify a legacy artifact as current without exercising the current
scoring contract.

## CRoMa is different

CRoMa requires every evaluation unit to be scoreable and fails rather than
pooling a partial subset, so it carries no support field. `CRoMaResult`, CRoMa
cache payloads, m-sweep rows, and per-sample rows also carry no aggregate
coverage field; their aligned values are finite for every requested evaluation
unit. We rejected compatibility aliases and partial CRoMa reporting because
they would preserve retired language or hide the effective denominator.

## Consequences

- Library consumers read `result.support` beside RI and MaRI and never infer a
  default.
- Benchmark consumers read one shared `support` and one shared set of cause
  columns.
- Legacy caches trigger recomputation; there are no aliases, conversion paths,
  or fallbacks.
- Released changelog entries continue to describe the schemas those releases
  actually shipped; the breaking migration is documented only under
  Unreleased.
