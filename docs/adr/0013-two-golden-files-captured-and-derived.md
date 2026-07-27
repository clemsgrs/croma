# Two golden files: one captured, one derived

The repository carries two golden metric files. They share a mechanism — a committed
JSON of numbers, compared against a fresh computation — but not their authority, and
the difference is deliberate.

This record governs the files that lock **croma's own metric values**, and only those.
`tests/fixtures/pathorob_apd_parity.json` reuses the mechanism but is not a third golden
file: it locks agreement with a *third party*, pairing accuracy matrices PathoROB
published with the APD values PathoROB published for them. Its authority is neither
captured nor derived but **lifted** — neither side of a pair may be computed by croma, so
a mismatch is evidence that the vendored reduction drifted from upstream rather than that
a snapshot went stale. It is named for parity rather than for goldenness to keep the two
kinds apart. See ADR-0011.

| | `fixtures/compute_golden_metrics.json` | `fixtures/library_golden_metrics.json` |
|---|---|---|
| Locks | the benchmark pipeline's output, end to end | metric values on the public API |
| Entry point | `scripts/bench/benchmark.py` | `croma.RI` / `MaRI` / `CRoMa` directly |
| `k` | chosen by the driver's `k-star` protocol | pinned |
| Authority | **captured** — whatever the pipeline produced | **derived** — argued from the geometry |
| Regenerable | yes, by re-running the driver | **no, on purpose** |
| Read by | `tests/test_compute_render_split.py` | `tests/test_library_golden.py` |

## Why the second file exists

Every number in the captured golden sits downstream of `k-star` auto-k selection, and
`scripts/bench/` is not shipped in the wheel. Two consequences follow.
A consumer who installs `croma` from PyPI has no value lock on the surface they actually
import. And a change to k-selection reshuffles the whole file whether or not a metric
moved — so the diff that should be the loudest signal in the repository is routinely
noisy for reasons unrelated to the metrics.

Extending the captured golden was the obvious cheaper move and was rejected: adding the
named embeddings to it would force them through `benchmark.py` and the `k-star` protocol,
reintroducing exactly the coupling the new file exists to remove. The two files have to
be entered by different doors.

## The derived-only rule

The library golden admits a value **only if that value follows from the construction of
the embedding**, and every entry carries the argument in a `derivation` field. Concretely,
on the named embeddings in `tests/metric_harness.py`:

- `confounder_dominant` — no sample has an SO neighbour in its top-k, so `so_total == 0`.
  `RI == 0.0`, and `MaRI == 0.0` for **any** `tau`, because the strictly positive weights
  multiply a zero numerator.
- `biology_dominant` — the mirror, `os_total == 0`, so `RI == MaRI == 1.0`.
- `contested` — by symmetry `so_total == os_total`, so `RI == MaRI == 0.5`; and
  `d_SO == d_OS` per sample, so `CRoMa == 0.0`.

The MaRI literals are therefore tau-independent, which is worth stating because auto-tau
is data-dependent: the test asserts them under auto-tau *and* under a deliberately
off-scale pinned tau, so a change to tau resolution cannot silently invalidate them.

**CRoMa on the two dominance embeddings is excluded.** Its value there is a function of
the specific within- and cross-cluster distances, not of anything the construction fixes.
Recording whatever the implementation happens to return would make the file a rubber stamp
for the code it is supposed to check — a green test that says only "it still does what it
did". Sign, bounds and the monotone ladder cover CRoMa on those embeddings instead, in
the property suite. `tests/test_library_golden.py` enforces the exclusion mechanically:
the only CRoMa entry admitted is `contested`'s `0.0`.

## No regeneration script

There is no generator and no `--update` flag for the library golden, and adding one would
defeat it. A regen flag turns a mismatch into a keystroke: the recorded numbers become
whatever the code last did, and the file stops being evidence. Without one, a mismatch has
to be argued — either the derivation is wrong, or the implementation is. That is the whole
value of the file, and it is affordable only because the file is small and the numbers are
literals a human can check by reading.

The captured golden keeps its regeneration path, which is the right trade for it: a full
row of pipeline outputs per model cannot be hand-derived, and re-running the driver is how
it is meant to be updated.

## Consequences

- **Do not merge the two files.** They answer different questions and grant different
  authority; a merged file would inherit the weaker rule.
- A new named embedding does not automatically earn an entry here. It earns one when a
  metric value on it can be derived; otherwise its coverage belongs in the property suite.
- The library golden's constants (`k`, `m`, confounder column, evaluation design) are
  asserted against `tests/metric_harness.py`, so moving `PINNED_K` fails loudly rather
  than leaving the file pinning a configuration nobody computes.
- ADR-0003's reproducibility posture is unaffected: both files are inputs to tests, not
  paper artifacts. Numbering skips 0010, which is local-only per ADR-0012.
