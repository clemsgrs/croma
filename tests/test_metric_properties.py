"""Property tests for the metric family, anchored to the definitions.

Every test drives the public ``compute`` through the adapter in ``metric_harness``, with
``k`` pinned to a single candidate. Nothing here reads from ``output/``, ``data/`` or
``paper/``: the geometry is synthetic and known by construction.
"""

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa, MaRI, RI
from metric_harness import (
    CONFOUNDER_COLUMN,
    METRICS,
    NAMED_EMBEDDINGS,
    PINNED_K,
    biology_dominant,
    compute_metric,
    confounder_dominant,
    constant_embedding,
    contested,
    isotropic_gaussian,
    metric_id,
    orthogonal_matrix,
)

_EMBEDDING_IDS = sorted(NAMED_EMBEDDINGS)
_EMBEDDINGS = [NAMED_EMBEDDINGS[name] for name in _EMBEDDING_IDS]

#: How each metric answers when ``label`` and the confounder change places. RI and MaRI are
#: ``SO / (SO + OS)``, so exchanging the two types reflects them about ``0.5``; CRoMa is the
#: signed margin ``(d_OS - d_SO) / (d_OS + d_SO)``, so the exchange flips its sign.
_ROLE_SWAP_IMAGE = {RI: lambda x: 1.0 - x, MaRI: lambda x: 1.0 - x, CRoMa: lambda x: -x}

#: The range each metric is defined on, as ``(low, high, strict)``. RI and MaRI are
#: proportions and attain their bounds (a neighbourhood with no OS evidence scores exactly
#: 1); CRoMa's bounds are open, since attaining one needs a typed distance of exactly zero.
_RANGE = {RI: (0.0, 1.0, False), MaRI: (0.0, 1.0, False), CRoMa: (-1.0, 1.0, True)}


def _role_swapped(manifest: pd.DataFrame) -> pd.DataFrame:
    """The same manifest with ``label`` and the confounder column exchanged.

    ``assign`` evaluates every keyword against the *original* frame, so this is a true
    swap rather than two sequential overwrites.
    """
    return manifest.assign(
        **{
            "label": manifest[CONFOUNDER_COLUMN],
            CONFOUNDER_COLUMN: manifest["label"],
        }
    )


def test_biology_dominant_neighbourhoods_are_all_so() -> None:
    """Every typed neighbour is SO, so ``os_total == 0`` and pooled RI is exactly 1."""
    features, manifest = biology_dominant()

    scored = compute_metric(RI, features, manifest)

    assert scored.score == pytest.approx(1.0)
    assert scored.sample_values.size > 0
    assert scored.sample_values == pytest.approx(1.0)


def test_confounder_dominant_neighbourhoods_are_all_os() -> None:
    """The mirror: every typed neighbour is OS, so ``so_total == 0`` and RI is exactly 0."""
    features, manifest = confounder_dominant()

    scored = compute_metric(RI, features, manifest)

    assert scored.score == pytest.approx(0.0)
    assert scored.sample_values.size > 0
    assert scored.sample_values == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [(RI, 0.5), (MaRI, 0.5), (CRoMa, 0.0)],
    ids=["RI", "MaRI", "CRoMa"],
)
def test_contested_sits_on_the_boundary(metric: type, expected: float) -> None:
    """One SO and one OS equidistant per sample puts every metric on its boundary value."""
    features, manifest = contested()

    scored = compute_metric(metric, features, manifest)

    assert scored.score == pytest.approx(expected)
    assert scored.sample_values.size > 0
    assert scored.sample_values == pytest.approx(expected)


@pytest.mark.parametrize("embedding", _EMBEDDINGS, ids=_EMBEDDING_IDS)
def test_named_embedding_keeps_k_below_its_scope_size(embedding) -> None:
    """The package's own sizing constraint: ``k`` strictly below the scope's row count."""
    features, manifest = embedding()

    assert len(features) == len(manifest)
    assert PINNED_K < len(manifest)


@pytest.mark.parametrize("embedding", _EMBEDDINGS, ids=_EMBEDDING_IDS)
@pytest.mark.parametrize("metric", METRICS, ids=metric_id)
def test_metric_is_invariant_under_an_orthogonal_transform(metric: type, embedding) -> None:
    """Cosine distance is preserved by an orthogonal ``Q``, so every metric must be too.

    This is the premise of comparing foundation models whose embedding spaces share no
    basis: a metric that moved under a rotation would be reading the coordinate system.
    Parametrized over every named embedding, so an embedding added later is covered here
    without touching this test.
    """
    features, manifest = embedding()
    rotated = features @ orthogonal_matrix(features.shape[1])

    scored = compute_metric(metric, features, manifest)
    scored_rotated = compute_metric(metric, rotated, manifest)

    # Without this the array comparison below would pass vacuously on an embedding where
    # the metric is undefined everywhere.
    assert scored.sample_values.size > 0
    assert scored_rotated.score == pytest.approx(scored.score, abs=1e-9)
    assert scored_rotated.sample_values == pytest.approx(scored.sample_values, abs=1e-9)


@pytest.mark.parametrize("embedding", _EMBEDDINGS, ids=_EMBEDDING_IDS)
@pytest.mark.parametrize("metric", METRICS, ids=metric_id)
def test_metric_is_antisymmetric_under_a_role_swap(metric: type, embedding) -> None:
    """Exchanging ``label`` with the confounder exchanges SO with OS, exactly.

    ``so_mask`` is ``same_label & ~same_center`` and ``os_mask`` its mirror, so a manifest
    whose two columns have changed places is scored on the same neighbours with the two
    typed roles swapped: ``RI -> 1 - RI``, ``MaRI -> 1 - MaRI``, ``CRoMa -> -CRoMa``. Any
    asymmetry in the masking, a sign error in CRoMa's numerator, or a weight applied to the
    wrong mask breaks this and nothing else in the suite would notice.

    ``k`` is pinned to a single candidate because k-selection maximises kNN balanced
    accuracy *on the labels*, which the swap changes -- a property of the selection rule,
    not of the metric. MaRI runs with auto-``tau`` on purpose: ``tau`` is resolved from the
    typed set ``same_label != same_center``, which is itself symmetric under the swap, so
    both sides resolve the identical temperature and the antisymmetry needs no pinning.
    """
    features, manifest = embedding()
    reflect = _ROLE_SWAP_IMAGE[metric]

    scored = compute_metric(metric, features, manifest)
    swapped = compute_metric(metric, features, _role_swapped(manifest))

    assert scored.sample_values.size > 0
    assert swapped.score == pytest.approx(reflect(scored.score), abs=1e-12)
    assert swapped.sample_values == pytest.approx(reflect(scored.sample_values), abs=1e-12)


@pytest.mark.parametrize(
    "embedding",
    # Deliberately not ``contested``: it is exactly symmetric by construction, so its SO and
    # OS neighbours sit at identical distances and their kNN ordering is genuinely ambiguous.
    # Permuting the rows reorders those ties, which is a property of the tie-break rather
    # than a violation of anything, and would make this test flaky for no signal. The two
    # dominance embeddings have strictly ordered top-k neighbours.
    [biology_dominant, confounder_dominant],
    ids=["biology_dominant", "confounder_dominant"],
)
@pytest.mark.parametrize("metric", METRICS, ids=metric_id)
def test_metric_is_invariant_under_a_row_permutation(metric: type, embedding) -> None:
    """Row order is not data: shuffling features and manifest together must change nothing.

    The score is unchanged and the per-sample values follow their rows. This is the property
    the occurrence-alignment machinery exists to preserve -- ``source_sample_index``,
    ``occurrence_defined_mask`` and the per-subset concatenation all assume a row keeps its
    identity through scoring -- and row order is exactly what a caller cannot be expected to
    hold fixed between an embedding matrix and the manifest that describes it.
    """
    features, manifest = embedding()
    permutation = np.random.default_rng(20260727).permutation(len(manifest))
    permuted_manifest = manifest.iloc[permutation].reset_index(drop=True)

    scored = compute_metric(metric, features, manifest)
    permuted = compute_metric(metric, features[permutation], permuted_manifest)

    # Every sample is defined on these embeddings, so ``sample_values`` is one value per row
    # in row order and the permutation below is a claim about rows rather than about a
    # filtered subsequence.
    assert scored.sample_values.size == len(manifest)
    assert permuted.score == pytest.approx(scored.score, abs=1e-12)
    assert permuted.sample_values == pytest.approx(scored.sample_values[permutation], abs=1e-12)


@pytest.mark.parametrize("embedding", _EMBEDDINGS, ids=_EMBEDDING_IDS)
@pytest.mark.parametrize("metric", METRICS, ids=metric_id)
def test_metric_stays_inside_its_range(metric: type, embedding) -> None:
    """RI and MaRI are proportions in ``[0, 1]``; CRoMa is a margin strictly inside ``(-1, 1)``.

    CRoMa's bounds are open because reaching them needs one of the two typed distances to be
    exactly zero, which is the degenerate case its NaN guard covers. Cheap, and it catches a
    saturating weight or a sign error on any fixture, pooled score and per-sample alike.
    """
    low, high, strict = _RANGE[metric]
    features, manifest = embedding()

    scored = compute_metric(metric, features, manifest)

    assert scored.sample_values.size > 0
    values = np.append(scored.sample_values, scored.score)
    if strict:
        assert np.all(values > low) and np.all(values < high)
    else:
        assert np.all(values >= low) and np.all(values <= high)


def test_a_constant_embedding_makes_croma_fail_rather_than_report_neutral() -> None:
    """Collapsed features make CRoMa's denominator zero, and zero is not a score.

    ``0.0`` would read as an exactly contested neighbourhood -- the metric's most confident
    statement of neutrality -- when in fact there is no neighbourhood to speak of: every
    distance is zero and the margin has no scale. ``croma.py`` guards this explicitly, and
    this is the only test that reaches the guard from a real embedding rather than by calling
    the private helper with hand-made arrays.
    """
    features, manifest = constant_embedding()

    with pytest.raises(RuntimeError, match=r"zero margin denominator .*d_OS \+ d_SO = 0"):
        compute_metric(CRoMa, features, manifest)


@pytest.mark.parametrize("metric", [RI, MaRI], ids=["RI", "MaRI"])
def test_a_constant_embedding_leaves_ri_and_mari_counting(metric: type) -> None:
    """The count-based pair survive the collapse, because neither divides by a distance.

    RI weights every neighbour by 1 and MaRI by ``exp(-0 / tau) = 1``, so both degrade to a
    count ratio over an arbitrary tie-break rather than to NaN. The value is not asserted --
    it is whatever the neighbour search's tie-break happens to return -- only that it is a
    proportion and that the collapse raises nothing.
    """
    features, manifest = constant_embedding()

    scored = compute_metric(metric, features, manifest)

    assert 0.0 <= scored.score <= 1.0
    assert scored.sample_values.size > 0


@pytest.mark.parametrize(
    ("metric", "expected", "tolerance"),
    [(RI, 0.5, 0.02), (MaRI, 0.5, 0.02), (CRoMa, 0.0, 0.01)],
    ids=["RI", "MaRI", "CRoMa"],
)
def test_structureless_features_score_at_the_null(
    metric: type, expected: float, tolerance: float
) -> None:
    """Isotropic Gaussian features encode neither label nor confounder, so nothing is said.

    SO and OS are equally available a priori, so RI and MaRI sit at ``0.5`` and CRoMa at
    ``0``. This is the paper's null anchor: the reading a model earns for having learned
    nothing, and the reference every reported score is a departure from.

    The one loose-tolerance test in an otherwise exact suite. The seed is pinned and the
    sample is large so the sampling error is far inside the tolerance. If this ever flakes,
    **delete it rather than widening it** -- a widened null anchor stops being an anchor, and
    a test people learn to ignore is worse than no test at all.
    """
    features, manifest = isotropic_gaussian()

    scored = compute_metric(metric, features, manifest)

    assert scored.score == pytest.approx(expected, abs=tolerance)
