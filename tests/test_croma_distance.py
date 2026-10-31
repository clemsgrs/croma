"""CRoMa's typed-neighbour distance: the cosine default and the Euclidean alternative.

Two kinds of reference are used, and neither is the library's own code path:

* an **exact** one, on a dyadic embedding whose every cosine distance is a multiple of
  ``1/16`` (see :func:`_dyadic_embedding`). At ``m`` a power of two every per-sample mean is
  exact too, so a correct implementation must return precisely the correctly rounded exact
  margin -- on any machine and any BLAS, because no intermediate is ever rounded;
* the **chord identity** on unit vectors, ``||a - b||^2 = 2 (1 - cos(a, b))``. At ``m = 1``
  it turns the cosine margin ``c = (r - 1)/(r + 1)``, ``r = d_OS / d_SO``, into the
  Euclidean margin ``e = (sqrt(r) - 1)/(sqrt(r) + 1)``, without touching which neighbour is
  which.
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


@pytest.mark.parametrize("distance", ["manhattan", "Cosine", "l2"])
def test_an_unknown_distance_is_refused_rather_than_silently_read_as_euclidean(distance):
    features, manifest = _gaussian_embedding(7)

    with pytest.raises(ValueError, match="distance"):
        _compute(features, manifest, distance=distance)
