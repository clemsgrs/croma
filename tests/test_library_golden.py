"""The library-level golden: values pinned because they can be derived, not captured.

``tests/fixtures/library_golden_metrics.json`` locks metric values on the public API --
``RI``/``MaRI``/``CRoMa`` called directly on the named embeddings from ``metric_harness``,
with ``k`` pinned. Nothing under ``scripts/`` is involved, so no number here sits downstream
of the benchmark driver's k-selection; the sibling golden
(``fixtures/compute_golden_metrics.json``, exercised by ``test_compute_render_split.py``)
covers that pipeline instead. ADR-0013 records why both files exist and how they differ.

The rule this file enforces is the *derived-only* one: an entry may be pinned only if its
value follows from the embedding's construction, and it must carry that argument in its
``derivation`` field. There is deliberately no regeneration script -- a mismatch is a claim
to be re-derived, not a number to be re-recorded -- so these tests also police the shape of
the file, not just its numbers.
"""

import json
from pathlib import Path

import pytest

from croma import CRoMa, MaRI, RI
from metric_harness import (
    CONFOUNDER_COLUMN,
    DEFAULT_M,
    EVALUATION_DESIGN,
    NAMED_EMBEDDINGS,
    PINNED_K,
    compute_metric,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "library_golden_metrics.json"

METRICS_BY_NAME = {"RI": RI, "MaRI": MaRI, "CRoMa": CRoMa}

#: A ``tau`` seven orders of magnitude above the typed-neighbour distances of the dominance
#: embeddings (auto-tau resolves to ~5e-5 there, ~1e-2 on ``contested``). At this temperature
#: ``exp(-d / tau)`` is ~1 for every neighbour, so the distance weighting is effectively
#: switched off -- which is exactly the perturbation the pinned MaRI values must survive.
#: Off-scale *upwards* on purpose: a tau far below the scale underflows the weights to zero
#: and makes MaRI undefined, which is a different claim from tau-independence.
OFF_SCALE_TAU = 1000.0

GOLDEN = json.loads(GOLDEN_PATH.read_text())
ENTRIES = GOLDEN["entries"]
ENTRY_IDS = [f"{entry['embedding']}-{entry['metric']}" for entry in ENTRIES]
MARI_ENTRIES = [entry for entry in ENTRIES if entry["metric"] == "MaRI"]
MARI_IDS = [f"{entry['embedding']}-MaRI" for entry in MARI_ENTRIES]


def _score(entry: dict, **kwargs) -> float:
    features, manifest = NAMED_EMBEDDINGS[entry["embedding"]]()
    scored = compute_metric(METRICS_BY_NAME[entry["metric"]], features, manifest, **kwargs)
    # Guards against a vacuous pass: a metric undefined on every sample has no support, and
    # its pooled score is not the derived value but an artefact of an empty reduction.
    assert scored.sample_values.size > 0
    return scored.score


def test_golden_pins_the_configuration_the_derivations_assume() -> None:
    """The derivations hold at a specific ``k``, ``m`` and scope; the file records all three.

    Without this the harness could move ``PINNED_K`` and leave the golden silently pinning
    values for a configuration nobody computes any more.
    """
    assert GOLDEN["k"] == PINNED_K
    assert GOLDEN["m"] == DEFAULT_M
    assert GOLDEN["confounder_column"] == CONFOUNDER_COLUMN
    assert GOLDEN["evaluation_design"] == EVALUATION_DESIGN


@pytest.mark.parametrize("entry", ENTRIES, ids=ENTRY_IDS)
def test_golden_value_matches_the_public_api(entry: dict) -> None:
    """The lock itself: the public API, called directly, still returns the pinned value."""
    assert _score(entry) == pytest.approx(entry["value"])


@pytest.mark.parametrize("entry", MARI_ENTRIES, ids=MARI_IDS)
def test_mari_golden_values_are_tau_independent(entry: dict) -> None:
    """Each MaRI entry is derived from a zero on one side of the ratio, so ``tau`` cancels.

    The weights ``exp(-d / tau)`` are strictly positive whatever ``tau`` is, so a zero
    numerator stays zero, a zero denominator-complement leaves the ratio at one, and equal
    SO/OS distances leave it at one half. Running the same assertion under auto-tau (above)
    and under a deliberately off-scale pinned tau is what turns that argument into a test:
    the value must not move when auto-tau does.
    """
    assert _score(entry, tau=OFF_SCALE_TAU, warn_tau=False) == pytest.approx(entry["value"])


@pytest.mark.parametrize("entry", ENTRIES, ids=ENTRY_IDS)
def test_every_entry_names_a_real_embedding_and_metric(entry: dict) -> None:
    """A typo in either field would otherwise silently drop an entry from the parametrization."""
    assert entry["embedding"] in NAMED_EMBEDDINGS
    assert entry["metric"] in METRICS_BY_NAME


@pytest.mark.parametrize("entry", ENTRIES, ids=ENTRY_IDS)
def test_every_entry_carries_its_derivation(entry: dict) -> None:
    """The derived-only rule, made mechanical: a number with no argument is not admissible."""
    assert entry["derivation"].strip()


def test_croma_is_pinned_only_where_it_is_derivable() -> None:
    """CRoMa's value on the dominance embeddings is geometry, not algebra -- so it stays out.

    On ``biology_dominant`` and ``confounder_dominant`` the margin depends on the actual
    within- and cross-cluster distances, which no derivation fixes; capturing whatever the
    implementation returns there would make this file a rubber stamp for the code it checks.
    ``contested`` is different: ``d_SO == d_OS`` per sample forces the margin to zero. Sign
    and bounds on the other two are covered by the property suite instead.
    """
    croma_entries = {
        entry["embedding"]: entry["value"] for entry in ENTRIES if entry["metric"] == "CRoMa"
    }

    assert croma_entries == {"contested": 0.0}
