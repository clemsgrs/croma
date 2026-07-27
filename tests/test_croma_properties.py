"""Property tests for CRoMa: search schedule, slide exclusion, response curve, formula.

CRoMa is the only metric of the three with non-trivial control flow -- it grows a neighbour
radius until every sample has found ``m`` SO and ``m`` OS neighbours -- and the only one whose
definition is a signed margin rather than a ratio. The four tests here pin the four things
that control flow could get wrong without any of the existing tests noticing: that the search
schedule is invisible in the answer, that same-slide neighbours never reach the margin, that
the margin responds to separation rather than merely to its sign, and that the whole chain
from features to score computes ``(d_OS - d_SO) / (d_OS + d_SO)``.

Every test drives the public ``CRoMa.compute`` through the adapter in ``metric_harness``. No
private helper is touched, so a refactor of the search internals is free to move anything as
long as the answers stay put.
"""

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa
from metric_harness import (
    CONFOUNDER_COLUMN,
    DEFAULT_M,
    compute_metric,
    toy_manifest,
    type_interleaved_ring,
)

#: The four label-confounder cells, in the order the two-cluster ring lays them out.
_CELLS = (("A", "V1"), ("A", "V2"), ("B", "V1"), ("B", "V2"))


def _two_cluster_ring(
    *, so_distance: float, os_distance: float, copies: int = 2
) -> tuple[np.ndarray, pd.DataFrame]:
    """Four cells on a circle, placed so every sample has the same ``d_SO`` and ``d_OS``.

    The cells sit at angles ``0``, ``a_SO``, ``a_OS``, ``a_OS + a_SO``, where
    ``a_SO = arccos(1 - so_distance)`` and ``a_OS = arccos(1 - os_distance)``. Reading the
    four pairwise gaps off that layout:

    * ``(A,V1)-(A,V2)`` and ``(B,V1)-(B,V2)`` are both ``a_SO`` apart -- the two SO pairs,
      at cosine distance ``so_distance``;
    * ``(A,V1)-(B,V1)`` and ``(A,V2)-(B,V2)`` are both ``a_OS`` apart -- the two OS pairs,
      at cosine distance ``os_distance``;
    * the remaining two pairs are OO, which CRoMa ignores.

    So at ``m = 1`` every one of the four samples has exactly the same margin, and the pooled
    median equals the per-sample value. That is what makes the fixture hand-checkable.

    Each cell is placed ``copies`` times at *identical* coordinates. The neighbour fetch caps
    at ``n_samples - 1`` counting the sample itself, so one neighbour is always out of reach;
    a second copy of every cell means the unreachable one is never the last of its type.
    Identical coordinates keep the distances exact, so the known answer stays exact too.
    """
    a_so = float(np.arccos(1.0 - float(so_distance)))
    a_os = float(np.arccos(1.0 - float(os_distance)))
    angles = [0.0, a_so, a_os, a_os + a_so]

    rows: list[list[float]] = []
    labels: list[str] = []
    confounders: list[str] = []
    for angle, (label, confounder) in zip(angles, _CELLS):
        for _ in range(int(copies)):
            rows.append([float(np.cos(angle)), float(np.sin(angle))])
            labels.append(label)
            confounders.append(confounder)
    return np.asarray(rows, dtype=float), toy_manifest(labels, confounders)


def _slide_trap(*, near_slide: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Eight tiles on a circle, with the query's nearest SO neighbour on ``near_slide``.

    Row 0 is the query ``(A,V1)`` on ``slide-q``. Row 1 is an ``(A,V2)`` tile a hair away from
    it -- the nearest SO neighbour by a wide margin -- and ``near_slide`` decides whether it
    shares the query's slide. Row 2 is a second ``(A,V2)`` tile, far away on its own slide.
    Row 3 is the query's ``(B,V1)`` OS neighbour, between the two. Rows 4-7 exist only so that
    every row has an SO and an OS neighbour of its own and no value comes back undefined.

    Built by hand rather than via ``toy_manifest``, whose one-tile-per-slide convention is
    exactly the thing under test here.
    """
    # Cosine distance from the query is ``1 - cos(angle)``: 0.01 to row 1, 0.5 to row 2,
    # 0.2 to row 3. The near SO neighbour is 50x closer than the far one.
    angles_deg = [
        0.0,
        float(np.rad2deg(np.arccos(0.99))),
        60.0,
        float(np.rad2deg(np.arccos(0.80))),
        120.0,
        150.0,
        200.0,
        250.0,
    ]
    cells = [
        ("A", "V1"),
        ("A", "V2"),
        ("A", "V2"),
        ("B", "V1"),
        ("B", "V2"),
        ("A", "V1"),
        ("B", "V1"),
        ("B", "V2"),
    ]
    slides = [
        "slide-q",
        near_slide,
        "slide-far",
        "slide-2",
        "slide-3",
        "slide-4",
        "slide-5",
        "slide-6",
    ]
    features = np.array(
        [[float(np.cos(np.deg2rad(a))), float(np.sin(np.deg2rad(a)))] for a in angles_deg],
        dtype=float,
    )
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(cells))],
            "image_path": [f"/tmp/{i}.png" for i in range(len(cells))],
            "label": [cell[0] for cell in cells],
            CONFOUNDER_COLUMN: [cell[1] for cell in cells],
            "slide_id": slides,
            "dataset": ["toy"] * len(cells),
        }
    )
    return features, manifest


def test_search_schedule_does_not_change_the_answer() -> None:
    """Any ``start_k`` and any ``k_growth_factor`` must give bit-identical output.

    The ``m`` nearest typed neighbours of a sample are a property of the embedding. The
    schedule only decides how many candidates the search looks at per pass, so it can change
    how much work is done and never what is found -- as long as every sample resolves. A
    sample that runs out of radius still unresolved comes back NaN, and *then* ``start_k``
    legitimately decides which samples those are; the invariant is false for a good reason.

    The precondition is therefore that no sample caps out, and it is asserted rather than
    assumed: every row of the manifest must come back with a finite value in both runs.
    ``type_interleaved_ring`` is sized for it -- its ``m``-th typed neighbours sit at a known
    offset that both runs reach with radius to spare.

    Note the knob: the adapter's ``k`` is CRoMa's ``start_k``, so the two runs differ in both
    the radius the search starts from and the factor it grows that radius by.
    """
    features, manifest = type_interleaved_ring()

    slow = compute_metric(CRoMa, features, manifest, k=1, m=DEFAULT_M, k_growth_factor=1.25)
    fast = compute_metric(CRoMa, features, manifest, k=25, m=DEFAULT_M, k_growth_factor=4.0)

    # Precondition: every row resolved, in both runs. Without this the equality below could
    # hold vacuously on two runs that both gave up in the same places.
    assert slow.sample_values.size == len(manifest)
    assert fast.sample_values.size == len(manifest)

    # Exact equality, not approximate: the two runs read the same distances off the same
    # embedding and average them in the same order, so there is nothing to round differently.
    assert fast.score == slow.score
    assert np.array_equal(fast.sample_values, slow.sample_values)


def test_same_slide_neighbour_is_excluded_from_the_margin() -> None:
    """CRoMa's own path applies the same-slide filter, not just the neighbour helper.

    A metric that let a same-slide tile count as a neighbour would read the query's own slide
    back to itself and report every model as more robust than it is. The two runs here share
    one embedding and differ in a single manifest cell -- the slide the near SO neighbour sits
    on -- which isolates the filter from every other moving part.

    With the near SO neighbour on the query's slide it must be skipped, leaving the far one:

        d_SO = 0.5, d_OS = 0.2 -> (0.2 - 0.5) / (0.2 + 0.5) = -0.3 / 0.7 = -3/7

    Move that same tile to its own slide and it becomes the nearest SO neighbour:

        d_SO = 0.01, d_OS = 0.2 -> (0.2 - 0.01) / (0.2 + 0.01) = 0.19 / 0.21 = +19/21

    A leak would not shade the answer, it would flip its sign: fragile reads as robust.
    """
    features, excluded = _slide_trap(near_slide="slide-q")
    _features, admitted = _slide_trap(near_slide="slide-other")

    with_filter = compute_metric(CRoMa, features, excluded, m=1)
    without_filter = compute_metric(CRoMa, features, admitted, m=1)

    # Every row is defined in both, so ``sample_values`` is in manifest order and row 0 is
    # the query.
    assert with_filter.sample_values.size == len(excluded)
    assert without_filter.sample_values.size == len(admitted)

    assert with_filter.sample_values[0] == pytest.approx(-3.0 / 7.0)
    assert without_filter.sample_values[0] == pytest.approx(19.0 / 21.0)


def test_croma_rises_monotonically_as_the_os_cluster_recedes() -> None:
    """Pushing the OS cluster away raises CRoMa strictly, and never off the ends of its range.

    ``d_SO`` is held at 0.5 while ``d_OS`` climbs across six separations, so the margin sweeps
    from confounder-dominant through the contested boundary into biology-dominant. A sign test
    would pass on a metric that saturated at the first step, clipped at the ends, or scaled by
    the wrong factor; a response curve does not.

    The range is open by construction -- ``(d_OS - d_SO) / (d_OS + d_SO)`` reaches ``+/-1``
    only when one of the two distances is zero -- so a value at or past an endpoint means the
    margin has been clipped or the denominator has lost a term.
    """
    separations = [0.1, 0.3, 0.6, 1.0, 1.4, 1.8]
    scores = [
        compute_metric(
            CRoMa,
            *_two_cluster_ring(so_distance=0.5, os_distance=separation),
            m=1,
        ).score
        for separation in separations
    ]

    assert len(scores) >= 5
    assert all(-1.0 < score < 1.0 for score in scores)
    assert all(lower < upper for lower, upper in zip(scores, scores[1:]))
    # The sweep crosses the boundary rather than creeping along one side of it, so a metric
    # stuck on a constant sign fails here too.
    assert scores[0] < 0.0 < scores[-1]


def test_m_one_margin_matches_the_documented_formula() -> None:
    """The whole chain -- features, neighbour search, score -- computes the documented margin.

    ``test_basic_margin`` pins the same formula against hand-made distance arrays, which
    proves the arithmetic and nothing about how the distances got there. This pins the answer
    to the *embedding*, so a neighbour search that picked the wrong typed neighbour, averaged
    over the wrong radius, or normalized the wrong way would show up here.

    Every sample in the fixture has one SO neighbour at cosine distance 0.2 and one OS
    neighbour at 0.8 (asserted below straight off the features, so the literal can be checked
    by reading), which makes the answer, per sample and pooled:

        (d_OS - d_SO) / (d_OS + d_SO) = (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    """
    features, manifest = _two_cluster_ring(so_distance=0.2, os_distance=0.8)

    # The two distances the literal is built from, read off the embedding itself. Rows 0, 2
    # and 4 are the first copy of (A,V1), (A,V2) and (B,V1): SO and OS partners of row 0.
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    assert 1.0 - float(unit[0] @ unit[2]) == pytest.approx(0.2)
    assert 1.0 - float(unit[0] @ unit[4]) == pytest.approx(0.8)

    scored = compute_metric(CRoMa, features, manifest, m=1)

    assert scored.sample_values.size == len(manifest)
    assert scored.sample_values == pytest.approx(0.6)
    assert scored.score == pytest.approx(0.6)
