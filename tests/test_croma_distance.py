"""CRoMa's typed-neighbour distance: the cosine default and the Euclidean and Manhattan
alternatives.

Three kinds of reference are used, and none is the library's own code path:

* an **exact** one, on a dyadic embedding whose every cosine distance is a multiple of
  ``1/16`` (see :func:`_dyadic_embedding`). At ``m`` a power of two every per-sample mean is
  exact too, so a correct implementation must return precisely the correctly rounded exact
  margin -- on any machine and any BLAS, because no intermediate is ever rounded;
* the **chord identity** on unit vectors, ``||a - b||^2 = 2 (1 - cos(a, b))``. At ``m = 1``
  it turns the cosine margin ``c = (r - 1)/(r + 1)``, ``r = d_OS / d_SO``, into the
  Euclidean margin ``e = (sqrt(r) - 1)/(sqrt(r) + 1)``, without touching which neighbour is
  which;
* margins worked **by hand** on five integer rows, whose Manhattan distances are integers,
  chosen so that Manhattan and cosine pick different nearest neighbours.

A distance computed elsewhere and handed to the search as a matrix is held to the library's
own ``"manhattan"`` path, itself pinned by the references above, on rows whose Manhattan
distances every implementation computes exactly.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa
from metric_harness import CONFOUNDER_COLUMN, toy_manifest

# --------------------------------------------------------------------------------------
# The dyadic embedding.
# --------------------------------------------------------------------------------------

_DIM = 64
_NONZEROS = 16
_LABELS = ("A", "B")
_CENTRES = ("C0", "C1", "C2")
_PER_CELL = 6


def _lcg(seed: int):
    """A fixed integer stream, so the embedding does not depend on numpy's RNG algorithms."""
    state = int(seed)
    while True:
        state = (6364136223846793005 * state + 1442695040888963407) % (1 << 64)
        yield state >> 33


def _dyadic_embedding() -> tuple[np.ndarray, pd.DataFrame]:
    """Rows with 16 nonzero entries of magnitude ``2**12`` or ``2**13``; norms ``2**14``/``2**15``.

    Each row draws 3-8 coordinates from its label's block and 3-8 from its centre's block,
    then pads to 16 with signed noise coordinates, so margins of both signs occur. Every
    norm is a power of two at least ``2**14`` -- large enough that the ``1e-12`` guard in the
    normalisation rounds away -- so every normalised entry is exactly ``+-1/4`` and every
    cosine distance is a multiple of ``1/16``. Distinct rows, so no distance is zero.
    """
    stream = _lcg(20261031)
    rows: list[list[int]] = []
    labels: list[str] = []
    centres: list[str] = []
    for li, label in enumerate(_LABELS):
        for ci, centre in enumerate(_CENTRES):
            for _ in range(_PER_CELL):
                label_block = list(range(8 * li, 8 * li + 8))
                centre_block = list(range(16 + 8 * ci, 16 + 8 * ci + 8))
                noise_block = list(range(40, _DIM))
                n_label = 3 + next(stream) % 6
                n_centre = 3 + next(stream) % 6
                chosen: dict[int, int] = {}
                for block, count in ((label_block, n_label), (centre_block, n_centre)):
                    for _ in range(count):
                        chosen[block.pop(next(stream) % len(block))] = 1
                while len(chosen) < _NONZEROS:
                    chosen[noise_block.pop(next(stream) % len(noise_block))] = (
                        1 if next(stream) % 2 else -1
                    )
                scale = 1 << (12 + next(stream) % 2)
                row = [0] * _DIM
                for coordinate, sign in chosen.items():
                    row[coordinate] = sign * scale
                rows.append(row)
                labels.append(label)
                centres.append(centre)
    if len({tuple(np.sign(r)) for r in rows}) != len(rows):
        raise AssertionError("dyadic embedding has duplicate directions; change the seed")
    return np.asarray(rows, dtype=np.float64), toy_manifest(labels, centres)


def _exact_cosine_margins(rows: np.ndarray, manifest: pd.DataFrame, m: int) -> list[Fraction]:
    """Brute-force per-sample CRoMa in exact rational arithmetic."""
    ints = [[int(v) for v in row] for row in rows]
    sq_norms = [sum(v * v for v in row) for row in ints]
    norms = [int(round(s**0.5)) for s in sq_norms]
    assert all(n * n == s for n, s in zip(norms, sq_norms))
    labels = manifest["label"].tolist()
    centres = manifest[CONFOUNDER_COLUMN].tolist()
    margins = []
    for i, a in enumerate(ints):
        so: list[Fraction] = []
        os_: list[Fraction] = []
        for j, b in enumerate(ints):
            if j == i:
                continue
            dot = sum(x * y for x, y in zip(a, b))
            d = 1 - Fraction(dot, norms[i] * norms[j])
            if labels[j] == labels[i] and centres[j] != centres[i]:
                so.append(d)
            elif labels[j] != labels[i] and centres[j] == centres[i]:
                os_.append(d)
        d_so = sum(sorted(so)[:m]) / m
        d_os = sum(sorted(os_)[:m]) / m
        margins.append((d_os - d_so) / (d_os + d_so))
    return margins


def _compute(features, manifest, **kwargs):
    return CRoMa.compute(features, manifest, confounder_column=CONFOUNDER_COLUMN, **kwargs)


# --------------------------------------------------------------------------------------
# The cosine default.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 4])
def test_default_distance_returns_the_exact_cosine_margin_of_every_sample(m):
    features, manifest = _dyadic_embedding()
    expected = [float(v) for v in _exact_cosine_margins(features, manifest, m)]

    result = _compute(features, manifest, m=m)

    assert result.sample_values_aligned.tolist() == expected
    assert result.f0 == float(np.mean(np.asarray(expected) <= 0.0))
    assert 0.0 < result.f0 < 1.0  # both signs occur, so the sign checks have teeth


def _gaussian_embedding(seed: int, *, shift: float = 0.0, per_cell: int = 7, dim: int = 24):
    """Seeded Gaussian rows with a label offset of ``1 + shift`` and a centre offset of ``1``.

    The last row is dropped so the row count is odd: the pooled median is then a sample
    value, and a monotone per-sample transform carries the median with it exactly.
    """
    rng = np.random.default_rng(seed)
    labels, centres, rows = [], [], []
    label_axes = rng.standard_normal((len(_LABELS), dim))
    centre_axes = rng.standard_normal((len(_CENTRES), dim))
    for li, label in enumerate(_LABELS):
        for ci, centre in enumerate(_CENTRES):
            for _ in range(per_cell):
                rows.append(
                    (1.0 + shift) * label_axes[li] + centre_axes[ci] + rng.standard_normal(dim)
                )
                labels.append(label)
                centres.append(centre)
    return np.asarray(rows[:-1]), toy_manifest(labels[:-1], centres[:-1])


def test_naming_the_cosine_default_changes_nothing():
    features, manifest = _gaussian_embedding(7)

    default = _compute(features, manifest, m=[1, 5])
    named = _compute(features, manifest, m=[1, 5], distance="cosine")

    for mm in (1, 5):
        assert np.array_equal(named[mm].sample_values_aligned, default[mm].sample_values_aligned)
        assert named[mm].value == default[mm].value


# --------------------------------------------------------------------------------------
# The Euclidean alternative.
# --------------------------------------------------------------------------------------


def _brute_force_euclidean_margins(rows: np.ndarray, manifest: pd.DataFrame, m: int) -> np.ndarray:
    """Per-sample margin from direct row differences, with no normalisation."""
    labels = manifest["label"].to_numpy()
    centres = manifest[CONFOUNDER_COLUMN].to_numpy()
    dist = np.sqrt(((rows[:, None, :] - rows[None, :, :]) ** 2).sum(axis=2))
    margins = []
    for i in range(len(rows)):
        others = np.arange(len(rows)) != i
        so = others & (labels == labels[i]) & (centres != centres[i])
        os_ = others & (labels != labels[i]) & (centres == centres[i])
        d_so = np.sort(dist[i, so])[:m].mean()
        d_os = np.sort(dist[i, os_])[:m].mean()
        margins.append((d_os - d_so) / (d_os + d_so))
    return np.asarray(margins)


@pytest.mark.parametrize("m", [1, 5])
def test_euclidean_distance_scores_the_raw_rows_without_normalising_them(m):
    features, manifest = _dyadic_embedding()
    expected = _brute_force_euclidean_margins(features, manifest, m)
    # The rows carry two different norms, so normalising first would move the margins.
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    assert np.max(np.abs(_brute_force_euclidean_margins(unit, manifest, m) - expected)) > 0.05

    result = _compute(features, manifest, m=m, distance="euclidean")

    np.testing.assert_allclose(result.sample_values_aligned, expected, rtol=0, atol=1e-12)


def _chord_margin(cosine_margin: np.ndarray) -> np.ndarray:
    """The m = 1 Euclidean margin implied by a cosine margin on unit vectors.

    With ``r = d_OS / d_SO = (1 + c) / (1 - c)`` and Euclidean distance ``sqrt(2 d)``, the
    Euclidean margin is ``(sqrt(r) - 1) / (sqrt(r) + 1)``.
    """
    c = np.asarray(cosine_margin, dtype=float)
    return (np.sqrt(1 + c) - np.sqrt(1 - c)) / (np.sqrt(1 + c) + np.sqrt(1 - c))


def test_euclidean_on_unit_vectors_is_the_chord_transform_of_cosine():
    shifts = (-0.6, -0.3, 0.0, 0.3, 0.6)
    cosine_medians, euclidean_medians = [], []
    for seed, shift in enumerate(shifts):
        features, manifest = _gaussian_embedding(100 + seed, shift=shift)
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)

        cosine = _compute(unit, manifest, m=1)
        euclidean = _compute(unit, manifest, m=1, distance="euclidean")

        c = cosine.sample_values_aligned
        e = euclidean.sample_values_aligned
        np.testing.assert_allclose(e, _chord_margin(c), rtol=0, atol=1e-9)
        assert np.array_equal(np.sign(e), np.sign(c))
        assert euclidean.f0 == cosine.f0
        cosine_medians.append(cosine.value)
        euclidean_medians.append(euclidean.value)

    assert len(set(np.sign(cosine_medians))) == 2  # the encoders straddle zero
    assert np.array_equal(np.argsort(euclidean_medians), np.argsort(cosine_medians))
    np.testing.assert_allclose(
        euclidean_medians, _chord_margin(np.asarray(cosine_medians)), rtol=0, atol=1e-9
    )


@pytest.mark.parametrize("m", [2, 5])
def test_euclidean_on_unit_vectors_keeps_the_cosine_neighbours(m):
    """Beyond m = 1 the margin is no closed-form transform, but the neighbours are the same.

    The reference picks each sample's ``m`` nearest SO and OS neighbours by *cosine*
    distance and only then converts each distance with the chord identity; the Euclidean
    search must land on exactly that margin.
    """
    features, manifest = _gaussian_embedding(11)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    labels = manifest["label"].to_numpy()
    centres = manifest[CONFOUNDER_COLUMN].to_numpy()
    cosine_distance = 1.0 - unit @ unit.T
    expected = []
    for i in range(len(unit)):
        others = np.arange(len(unit)) != i
        typed = []
        for mask in (
            others & (labels == labels[i]) & (centres != centres[i]),
            others & (labels != labels[i]) & (centres == centres[i]),
        ):
            nearest = np.sort(cosine_distance[i, mask])[:m]
            typed.append(np.sqrt(2.0 * nearest).mean())
        d_so, d_os = typed
        expected.append((d_os - d_so) / (d_os + d_so))

    result = _compute(unit, manifest, m=m, distance="euclidean")

    np.testing.assert_allclose(result.sample_values_aligned, expected, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------------------
# The Manhattan alternative.
# --------------------------------------------------------------------------------------

#: Five hand-built rows of norm 15, so they are unit vectors up to one common scale and every
#: Manhattan distance is an integer. ``a``'s two SO candidates are ``b`` and ``c``: by cosine
#: ``c`` is nearer (``1/3`` against ``2/5``), by Manhattan ``b`` is (``18`` against ``20``).
_HAND_ROWS = {
    "a": ((15, 0, 0), "A", "C0"),
    "b": ((9, 12, 0), "A", "C1"),
    "c": ((10, 5, 10), "A", "C1"),
    "o": ((0, 0, 15), "B", "C0"),
    "p": ((0, 15, 0), "B", "C1"),
}


def _hand_built() -> tuple[np.ndarray, pd.DataFrame]:
    rows, labels, centres = zip(*_HAND_ROWS.values())
    return np.asarray(rows, dtype=np.float64), toy_manifest(list(labels), list(centres))


def test_manhattan_distance_sums_absolute_coordinate_differences():
    features, manifest = _hand_built()
    # By hand, (d_OS - d_SO) / (d_OS + d_SO) with d = sum_j |x_j - y_j|:
    #   a: SO {b: 18, c: 20} -> 18, OS {o: 30}         -> 12/48
    #   b: SO {a: 18},          OS {p: 12}             -> -6/30
    #   c: SO {a: 20},          OS {p: 30}             -> 10/50
    #   o: SO {p: 30},          OS {a: 30}             -> 0
    #   p: SO {o: 30},          OS {b: 12, c: 30} -> 12 -> -18/42
    expected = [Fraction(12, 48), Fraction(-6, 30), Fraction(10, 50), 0, Fraction(-18, 42)]

    result = _compute(features, manifest, m=1, distance="manhattan")

    assert result.sample_values_aligned.tolist() == [float(v) for v in expected]


def test_manhattan_can_choose_a_different_nearest_neighbour_than_cosine():
    features, manifest = _hand_built()
    anchor = list(_HAND_ROWS).index("a")

    cosine = _compute(features, manifest, m=1).sample_values_aligned[anchor]
    manhattan = _compute(features, manifest, m=1, distance="manhattan").sample_values_aligned[
        anchor
    ]

    # Cosine reaches SO through c (d_SO = 1/3, d_OS = 1); through b it would be 3/7.
    assert cosine == pytest.approx(1 / 2, abs=1e-12)
    # Manhattan reaches SO through b (d_SO = 18, d_OS = 30); through c it would be 1/5.
    assert manhattan == 1 / 4


@pytest.mark.parametrize("m", [1, 5])
def test_manhattan_distance_scores_the_raw_rows_without_normalising_them(m):
    features, manifest = _dyadic_embedding()
    labels = manifest["label"].to_numpy()
    centres = manifest[CONFOUNDER_COLUMN].to_numpy()

    def margins(rows):
        dist = np.abs(rows[:, None, :] - rows[None, :, :]).sum(axis=2)
        out = []
        for i in range(len(rows)):
            others = np.arange(len(rows)) != i
            d_so = np.sort(dist[i, others & (labels == labels[i]) & (centres != centres[i])])
            d_os = np.sort(dist[i, others & (labels != labels[i]) & (centres == centres[i])])
            d_so, d_os = d_so[:m].mean(), d_os[:m].mean()
            out.append((d_os - d_so) / (d_os + d_so))
        return np.asarray(out)

    expected = margins(features)
    # The rows carry two different norms, so normalising first would move the margins.
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    assert np.max(np.abs(margins(unit) - expected)) > 0.05

    result = _compute(features, manifest, m=m, distance="manhattan")

    np.testing.assert_allclose(result.sample_values_aligned, expected, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------
# A distance computed elsewhere, handed to the search as a matrix.
# --------------------------------------------------------------------------------------


def _l1_matrix(rows: np.ndarray) -> np.ndarray:
    return np.abs(rows[:, None, :] - rows[None, :, :]).sum(axis=2)


def _dyadic_on_shared_slides() -> tuple[np.ndarray, pd.DataFrame]:
    """The dyadic embedding with every two consecutive rows (one cell) on one slide."""
    features, manifest = _dyadic_embedding()
    manifest = manifest.assign(group_id=[f"slide-{i // 2}" for i in range(len(manifest))])
    return features, manifest


def _hand_built_in_two_subsets() -> tuple[np.ndarray, pd.DataFrame]:
    """The hand-built rows as one 2x2 subset, and the same rows doubled as another."""
    features, manifest = _hand_built()
    rows, labels, centres = zip(*_HAND_ROWS.values())
    paired = toy_manifest(list(labels) * 2, list(centres) * 2)
    paired["subset"] = ["p0"] * len(rows) + ["p1"] * len(rows)
    return np.vstack([features, 2.0 * features]), paired


_EXACT_L1_COHORTS = {
    "hand-built": _hand_built,
    "dyadic": _dyadic_embedding,
    "dyadic-shared-slides": _dyadic_on_shared_slides,
}


@pytest.mark.parametrize(
    "cohort, m",
    [
        ("hand-built", 1),
        ("dyadic", 1),
        ("dyadic", 5),
        ("dyadic-shared-slides", 1),
        ("dyadic-shared-slides", 5),
    ],
)
def test_a_precomputed_manhattan_matrix_scores_exactly_as_the_manhattan_distance(cohort, m):
    # Every distance in these cohorts is an exact sum, and many are tied, so the two paths
    # agree only if the matrix path keeps every neighbour, exclusion and tie the same.
    features, manifest = _EXACT_L1_COHORTS[cohort]()

    reference = _compute(features, manifest, m=m, distance="manhattan")
    precomputed = _compute(features, manifest, m=m, distance=_l1_matrix)

    assert np.array_equal(
        precomputed.sample_values_aligned, reference.sample_values_aligned, equal_nan=True
    )
    assert (precomputed.value, precomputed.f0, precomputed.ltm_alpha) == (
        reference.value,
        reference.f0,
        reference.ltm_alpha,
    )


def test_a_distance_callable_is_handed_each_subsets_rows_as_given():
    features, manifest = _hand_built_in_two_subsets()
    seen: list[np.ndarray] = []

    def l1(rows: np.ndarray) -> np.ndarray:
        seen.append(rows.copy())
        return _l1_matrix(rows)

    result = _compute(features, manifest, m=1, evaluation_design="paired_2x2", distance=l1)
    reference = _compute(
        features, manifest, m=1, evaluation_design="paired_2x2", distance="manhattan"
    )

    assert [s.tolist() for s in seen] == [features[:5].tolist(), features[5:].tolist()]
    assert np.array_equal(result.sample_values_aligned, reference.sample_values_aligned)
    # Doubling every row doubles every distance, so the second subset's margins are the first's.
    assert result.sample_values_aligned[:5].tolist() == result.sample_values_aligned[5:].tolist()


@pytest.mark.parametrize(
    "matrix, match",
    [
        (lambda rows: _l1_matrix(rows)[:, :-1], "square"),
        (lambda rows: _l1_matrix(rows) + 1.0, "diagonal"),
        (lambda rows: -_l1_matrix(rows), "negative"),
        (lambda rows: np.where(np.eye(len(rows), dtype=bool), 0.0, np.nan), "finite"),
    ],
    ids=["not-square", "nonzero-diagonal", "negative", "not-finite"],
)
def test_a_distance_callable_must_return_a_distance_matrix(matrix, match):
    features, manifest = _hand_built()

    with pytest.raises(ValueError, match=match):
        _compute(features, manifest, m=1, distance=matrix)


@pytest.mark.parametrize("distance", ["l1", "Cosine", "l2"])
def test_an_unknown_distance_is_refused_rather_than_silently_read_as_euclidean(distance):
    features, manifest = _gaussian_embedding(7)

    with pytest.raises(ValueError, match="distance"):
        _compute(features, manifest, distance=distance)
