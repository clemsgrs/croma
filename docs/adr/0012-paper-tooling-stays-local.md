# Paper tooling is tracked after manuscript publication

The manuscript is public, so the reversal condition in the original version of this ADR has
occurred. Paper generators, their tests, and the machine-readable model metadata are tracked in
the repository. This restores ADR-0003's reproducibility posture: a reader can inspect the
generators together with the public manuscript artifacts they produce.

The paper assembly remains outside the Python package. `scripts/repro/` and tests that require
paper-only inputs are excluded from the source distribution without exception. The canonical
model registry lives at `scripts/bench/model_metadata.csv`, because benchmark plotting consumes
its structured family, tone, and order fields at runtime (ADR-0005 and ADR-0017); paper
generators read that same benchmark-owned source.

## Why

Before the manuscript was public, tracking a build system for an unavailable document offered
readers no reproducibility benefit. The original decision therefore kept `scripts/repro/`, its
tests, and ADR-0010 local until the paper could be read and checked. That temporary publication
boundary no longer applies.

Repository tracking and package distribution have different audiences. The repository is the
auditable home for the paper workflow; the package ships the metrics library and benchmark
tools. Excluding paper assembly from the source distribution keeps `pip`-facing artifacts
focused and avoids tests whose ignored run outputs are unavailable in a clean package build.
Keeping shared runtime data under `scripts/bench/` prevents the plotting code from reaching
across that boundary or acquiring a second, hard-coded identity map.

## Shape

**Tracked in the repository:**

- `scripts/repro/`, including `paper_manifest.py` and templates
- the corresponding tests under `tests/`
- ADR-0010 and the other presentation ADRs

**Excluded from the source distribution:**

- paper generators under `scripts/repro/`
- paper-only tests and ADR-0010
- all paper assembly artifacts not otherwise published

**Included benchmark runtime data:**

- `scripts/bench/model_metadata.csv`, the single model registry used by benchmark plots and
  paper generators

## Consequences

- Paper-tool changes are reviewed and tested like other repository code.
- Tests requiring ignored benchmark outputs remain local pre-flight checks by construction.
- Model provenance and plot identity stay machine-readable and single-sourced without making
  the paper assembly part of the installed `croma` library.

## Historical decision

The original ADR was titled “Paper tooling stays local until the manuscript is published.” Its
explicit reversal condition was manuscript publication. This revision records that transition
rather than silently deleting the reason the tooling was once untracked.
