"""Two properties of MaRI that the manuscript's account of MaRI versus RI rests on.

**The pooling identity.** RI and MaRI both pool as a ratio of sums over every typed (SO/OS)
neighbour slot, and MaRI's weights ``exp(-d / tau)`` are not normalised per anchor. So the
gap between their logits is exactly the log ratio of the mean SO weight to the mean OS weight,
pooled over all slots of all anchors::

    logit(MaRI) - logit(RI) = log(w_SO / w_OS)

A per-anchor normalisation, or an average of per-anchor ratios, would break it. The test
checks the library's pooled scores against weights computed here by brute force.

**Scale invariance under the automatic tau.** Resolved as the median typed-neighbour distance,
``tau`` scales with the distances, so multiplying every distance by ``c > 0`` leaves every
weight -- and so MaRI -- unchanged. A pinned ``tau`` does not move with the distances, so the
same rescaling sharpens or flattens its weights. This is tested at the level of the
neighbour-scoring function, where distances can be rescaled directly. Rescaling features cannot
do it, because cosine distance ignores feature scale.
"""

import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI
from croma.metrics.neighbors import _prepare_neighbors

K = 5
TAU = 0.15


def _synthetic(seed: int = 7, per_cell: int = 12, dim: int = 6):
    """Two labels x two confounders, each cell a jittered cluster.

    The label and confounder centroids are separated by comparable amounts, so every
    neighbourhood mixes SO and OS neighbours. Groups are pairs of rows, which exercises the
    same-group exclusion.
    """
    rng = np.random.default_rng(seed)
    label_axis = rng.standard_normal(dim)
    confounder_axis = rng.standard_normal(dim)
    rows, labels, confounders = [], [], []
    for label in (0, 1):
        for confounder in (0, 1):
            centre = 3.0 + 0.6 * label * label_axis + 0.5 * confounder * confounder_axis
            rows.append(centre + 0.35 * rng.standard_normal((per_cell, dim)))
            labels += [f"L{label}"] * per_cell
            confounders += [f"C{confounder}"] * per_cell
    features = np.vstack(rows)
    n = len(features)
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tmp/{i}.png" for i in range(n)],
            "group_id": [f"g{i // 2}" for i in range(n)],
            "label": labels,
            "center": confounders,
            "dataset": ["toy"] * n,
        }
    )
    return features, manifest


def _brute_force_typed_weights(features, manifest, k, tau):
    """Mean SO and OS weight over all typed slots, from an exhaustive cosine neighbour scan."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    distances = 1.0 - unit @ unit.T
    labels = manifest["label"].to_numpy()
    centres = manifest["center"].to_numpy()
    groups = manifest["group_id"].to_numpy()
    so, os_ = [], []
    for i in range(len(features)):
        candidates = np.flatnonzero(groups != groups[i])
        nearest = candidates[np.argsort(distances[i, candidates], kind="stable")[:k]]
        for j in nearest:
            weight = np.exp(-distances[i, j] / tau)
            same_label, same_centre = labels[j] == labels[i], centres[j] == centres[i]
            if same_label and not same_centre:
                so.append(weight)
            elif same_centre and not same_label:
                os_.append(weight)
    return np.mean(so), np.mean(os_)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def test_logit_gap_between_mari_and_ri_is_the_log_mean_weight_ratio() -> None:
    features, manifest = _synthetic()
    kwargs = dict(confounder_column="center", k_candidates=[K])

    ri = RI.compute(features, manifest, **kwargs).value
    mari = MaRI.compute(features, manifest, tau=TAU, warn_tau=False, **kwargs).value
    w_so, w_os = _brute_force_typed_weights(features, manifest, K, TAU)

    assert 0.0 < ri < 1.0 and mari != pytest.approx(ri)
    assert _logit(mari) - _logit(ri) == pytest.approx(np.log(w_so / w_os), abs=1e-10)


def _prepared_neighbours():
    features, manifest = _synthetic(seed=11)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    idx, dist, valid = _prepare_neighbors(unit, manifest["group_id"].to_numpy(), K)
    labels = pd.factorize(manifest["label"])[0]
    centres = pd.factorize(manifest["center"])[0]
    return labels, centres, idx, dist, valid


def _mari_from_neighbours(labels, centres, idx, dist, valid, tau: float) -> float:
    scored = MaRI._score_all_k_from_neighbors(labels, centres, idx, dist, valid, [K], tau=tau)
    return float(scored[K].pooled)


def _auto_tau(labels, centres, idx, dist, valid) -> float:
    """The automatic tau: the median typed-neighbour distance at the operating k."""
    typed = MaRI._typed_neighbor_distances_from_neighbors(
        labels=labels, centers=centres, neigh_idx=idx, neigh_dist=dist, valid_counts=valid, k=K
    )
    return float(np.median(typed))


@pytest.mark.parametrize("scale", [0.1, 3.7])
def test_rescaling_distances_leaves_mari_unchanged_at_the_automatic_tau(scale: float) -> None:
    labels, centres, idx, dist, valid = _prepared_neighbours()
    scaled = dist * scale

    original = _mari_from_neighbours(
        labels, centres, idx, dist, valid, _auto_tau(labels, centres, idx, dist, valid)
    )
    rescaled = _mari_from_neighbours(
        labels, centres, idx, scaled, valid, _auto_tau(labels, centres, idx, scaled, valid)
    )

    assert rescaled == pytest.approx(original, rel=1e-12)


@pytest.mark.parametrize("scale", [0.1, 3.7])
def test_rescaling_distances_moves_mari_at_a_pinned_tau(scale: float) -> None:
    labels, centres, idx, dist, valid = _prepared_neighbours()
    pinned = _auto_tau(labels, centres, idx, dist, valid)

    original = _mari_from_neighbours(labels, centres, idx, dist, valid, pinned)
    rescaled = _mari_from_neighbours(labels, centres, idx, dist * scale, valid, pinned)

    assert abs(rescaled - original) > 1e-3
