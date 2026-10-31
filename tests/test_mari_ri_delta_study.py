"""The paired cluster bootstrap of Delta = MaRI - RI, on fixed neighbours.

Worked by hand from five samples whose top-2 neighbours are written out literally (scoring
reads only each neighbour's label, confounder and distance, so the lists need not be
geometrically realisable):

    anchor  cell    top-2 neighbours (type, distance)
    0       L0 C0   s1 SO 0.1, s2 OS 0.4
    1       L0 C1   s0 SO 0.2
    2       L1 C0   s0 OS 0.2, s1 OO 0.5
    3       L1 C1   (none)
    4       L0 C0   (none)

The typed slots are SO 0.1, OS 0.4 (anchor 0), SO 0.2 (anchor 1) and OS 0.2 (anchor 2). With
every anchor drawn once RI = 2 / 4 and tau, the median typed distance, is 0.2.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import mari_ri_delta as study  # noqa: E402
import mari_vs_ri  # noqa: E402

E = np.exp
NEIGHBOURS = mari_vs_ri.Neighbours(
    labels=np.array([0, 0, 1, 1, 0]),
    centers=np.array([0, 1, 0, 1, 0]),
    neigh_idx=np.array([[1, 2], [0, -1], [0, 1], [-1, -1], [-1, -1]]),
    neigh_dist=np.array([[0.1, 0.4], [0.2, np.inf], [0.2, 0.5], [np.inf, np.inf], [np.inf] * 2]),
    valid_counts=np.array([2, 1, 2, 0, 0]),
)
K = 2


def test_every_anchor_once_reproduces_the_library_scoring():
    slots = study.typed_slots(NEIGHBOURS, K)
    ri, mari, tau = study.replicate_scores(slots, np.ones(5))
    assert ri == 0.5
    assert tau == 0.2
    assert mari == pytest.approx((E(-0.5) + E(-1)) / (E(-0.5) + E(-2) + E(-1) + E(-1)), rel=1e-12)
    # ...which is what the library's own scoring and automatic tau give on these neighbours.
    assert tau == mari_vs_ri.automatic_tau(NEIGHBOURS, K)
    assert ri == mari_vs_ri._score(study.RI, NEIGHBOURS, K).pooled
    library = mari_vs_ri._score(study.MaRI, NEIGHBOURS, K, tau=tau).pooled
    assert mari == pytest.approx(library, rel=1e-12)


def test_a_drawn_anchor_counts_once_per_draw_and_tau_is_re_resolved():
    slots = study.typed_slots(NEIGHBOURS, K)
    # Anchor 0 twice and anchor 2 once: typed distances 0.1, 0.1, 0.2, 0.4, 0.4.
    ri, mari, tau = study.replicate_scores(slots, np.array([2, 0, 1, 0, 0]))
    assert ri == 2 / 5
    assert tau == 0.2
    assert mari == pytest.approx(2 * E(-0.5) / (2 * E(-0.5) + 2 * E(-2) + E(-1)), rel=1e-12)
    # Anchor 0 alone: an even count, so tau is the mean of the middle pair (0.1 + 0.4) / 2.
    ri, mari, tau = study.replicate_scores(slots, np.array([1, 0, 0, 0, 0]))
    assert (ri, tau) == (0.5, 0.25)
    assert mari == pytest.approx(E(-0.4) / (E(-0.4) + E(-1.6)), rel=1e-12)


def test_delta_summary_reports_the_interval_and_whether_it_excludes_zero():
    ri = np.full(5, 0.5)
    mari = np.array([0.5, 0.625, 0.75, 0.875, 0.9375])
    row = study.summarise(0.5, 0.75, ri, mari, level=0.5)
    assert row["delta"] == 0.25
    assert (row["delta_lo"], row["delta_hi"]) == (0.125, 0.375)
    assert row["excludes_zero"]
    assert row["logit_gap"] == pytest.approx(np.log(3.0), rel=1e-12)
    row = study.summarise(0.5, 0.75, ri, np.array([0.25, 0.375, 0.5, 0.625, 0.75]), level=0.5)
    assert not row["excludes_zero"]
