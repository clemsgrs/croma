"""Property tests for the metric family, anchored to the definitions.

Every test drives the public ``compute`` through the adapter in ``metric_harness``, with
``k`` pinned to a single candidate. Nothing here reads from ``output/``, ``data/`` or
``paper/``: the geometry is synthetic and known by construction.
"""

import pytest

from croma import CRoMa, MaRI, RI
from metric_harness import (
    METRICS,
    NAMED_EMBEDDINGS,
    PINNED_K,
    biology_dominant,
    compute_metric,
    confounder_dominant,
    contested,
    metric_id,
    orthogonal_matrix,
)

_EMBEDDING_IDS = sorted(NAMED_EMBEDDINGS)
_EMBEDDINGS = [NAMED_EMBEDDINGS[name] for name in _EMBEDDING_IDS]


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
