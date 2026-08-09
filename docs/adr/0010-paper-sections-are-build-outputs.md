# Every number in `paper/sections/` is generated, never typed

Any quantity in the manuscript that can be derived from a persisted artifact — a
benchmark `metrics.csv` under `output/`, or `scripts/bench/model_metadata.csv` — is
emitted by a generator. This holds for table bodies, for the scalars in
`generated_values.tex`, **and for captions**. A `.tex` file under `paper/sections/`
that a generator owns is a build output: it is rebuilt from a single manifest by a
single entrypoint, it is never hand-edited, and a test fails when it drifts.

The manifest (`scripts/repro/paper_manifest.py`) is the one place that maps a
benchmark to its protocol, its `metrics.csv`, its output `.tex`, and its caption. The
model roster is read from the `metrics.csv`, never declared: a roster constant is a
second source of truth for what the run contained.

Caption prose stays authorial. Every *number* and every *model-name list* inside it is
a computed placeholder, and every caption sentence that asserts something about the
data carries a predicate the generator evaluates. A caption claim that has stopped
being true raises, rather than shipping.

## Why

Before this, the pipeline that produced the results and the pipeline that produced the
paper were disjoint, and nothing connected them.

`run_benchmarks.sh` — the script that actually ran the 21-model sweep — wrote
`output/metrics/median-k/` and rendered plots. It never touched `paper/sections/`.
The tables were written by `reproduce_faithful.py`, which pinned a 16-model roster in
a string constant, ran at protocol `k-star` while the paper reported `median-k`, and
exempted the main table entirely (`out_tex=None`) so that its hand-written caption
would survive. Two more tables (prostate, panda) had no generator invocation anywhere
in the repo.

The result: `tab:main-results`, cited nineteen times, was five models, one protocol and
one operating point out of date. Its caption claimed the models were read "at the
shared operating point $k{=}15$" when the live run used $k{=}11$; it reported a support
span of $11.9\%$–$53.2\%$ against a live $10\%$–$46\%$; and it asserted that "no model
reaches two-thirds coverage" while a model in the run held $68.0\%$.

That last one is the argument for generating captions. Captions are where the rot
concentrates, because captions are where the *claims* live. A hand-written caption is a
claim about the data with no test attached, sitting next to a table that has one. Every
stale number this audit turned up was in a caption or in a body no generator owned.

`paper/` is git-ignored (ADR-0003), so none of this drift is visible in `git status` or
in review. The freshness test is therefore not a nicety; it is the only backstop.

## Considered options

- **Generate bodies, keep captions hand-written.** The status quo, and what
  `generate_model_tables.py` does today. It is defensible *there*: the model-summary and
  model-provenance captions carry no run-derived numbers, so a template with a
  `%%BODY%%` hole is safe. It is not defensible for the results tables, whose primary
  caption carries seven derived quantities and three assertions about the data.

- **Generate captions as free prose.** Rejected. A caption's voice is authorial and
  worth keeping under human control. The generator should own the numbers, not the
  sentences. Hence the placeholder-and-predicate split: the writer keeps the sentence,
  the generator keeps every figure inside it and refuses to print a sentence whose
  predicate is false.

- **Track `paper/` in git so drift shows up in `git diff`.** Rejected separately by
  ADR-0003 (the tree carries a large figure corpus). The freshness test recovers the
  check without the commit.

- **Let each generator keep its own path constants.** Rejected: this is precisely how
  `_MEDIAN`, `_FAITHFUL` and `CFG` came to disagree with each other about which
  protocol the paper reports. Three lists, hand-maintained, no cross-check.

## Consequences

- Editing a number in a caption by hand now fails a test. This is correct — the number
  belongs to the run, not to the sentence.
- A caption claim that ceases to hold fails the build instead of shipping. This is the
  point of the exercise, not a side effect.
- Adding a benchmark means adding one manifest row, not editing three lists and
  remembering two CLI invocations.
- The operating-point phrase is computed from the run's own `k` column, so a `k-star`
  table can no longer describe itself as having a shared `k` (the bug in the PANDA
  caption, which claimed a shared $k{=}9$ for a run whose models sit at $k \in \{3, 9\}$).
- The natural-image control's placement is a property of the generator, not of a
  hand-edit: it is emitted in its own band below the ranked panel, and it is excluded
  from sorting and from the per-column bold. Previously this was a hand-edit that no
  generator knew about, which is how the control came to falsify a caption claim.
- `generate_model_tables.py` keeps its `%%BODY%%` template split, but joins the same
  entrypoint. Its captions do quote metadata-derived figures (`307`M nominal parameters,
  the measured `303.3`–`303.9`M range) which are typed rather than computed. That is the
  same hazard in a milder form, and remains a follow-up.
