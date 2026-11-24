"""Shared load scaffold for the Camelyon-faithful neighbourhood-composition experiments.

``contested_fraction_experiment.py``, ``oo_fraction_experiment.py``,
``support_vs_k_experiment.py``, ``tau_sensitivity_experiment.py`` and
``typed_neighbor_rank_experiment.py`` each repeated the same three preliminaries: the
repo ``sys.path`` bootstrap, factorising the manifest's label/confounder/group columns,
and loading + L2-normalising a model's embedding matrix. Those live here now.

Each experiment keeps its own per-model *sweep* -- they genuinely differ (some call
``croma.metrics.neighbors._prepare_neighbors``, some build a full cosine-distance matrix,
one calls ``RI``/``MaRI.compute``) -- so only the shared load steps are factored out.
Load-bearing choices that differ between experiments (float32 vs float64 embeddings,
whether to normalise, and the two metadata dtype conventions) are exposed as parameters
so every call is byte-for-byte equivalent to the inline code it replaces.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_meta(df: pd.DataFrame, *, compact: bool = False):
    """Factorise the manifest into (labels, confounder, group) code arrays.

    ``compact=False`` (the ``_prepare_neighbors`` experiments): int64 label/confounder
    codes and the raw ``group_id`` strings, matching ``_prepare_neighbors``'s expected
    input. ``compact=True`` (the full distance-matrix experiments): int16 label/confounder
    codes and int32 group codes for the argsort scan.
    """
    if compact:
        labels = pd.factorize(df["label"])[0].astype(np.int16)
        conf = pd.factorize(df["confounder"])[0].astype(np.int16)
        group = pd.factorize(df["group_id"])[0].astype(np.int32)
    else:
        labels = pd.factorize(df["label"])[0].astype(int)
        conf = pd.factorize(df["confounder"])[0].astype(int)
        group = df["group_id"].astype(str).to_numpy()
    return labels, conf, group


def prepare_embedding(X, dtype=np.float64, *, normalize: bool = True) -> np.ndarray:
    """Cast (and optionally L2-normalise) an in-memory embedding matrix.

    ``dtype`` (float32 vs float64) is load-bearing -- it changes the downstream cosine
    geometry -- so it stays explicit at the call site. Use this on the row-selected
    matrix returned by ``views.BenchmarkView.features`` (which yields the raw stored
    rows) to reproduce what ``load_embedding`` did for a whole-file matrix.
    """
    X = np.asarray(X).astype(dtype)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


def load_embedding(path, dtype=np.float64, *, normalize: bool = True) -> np.ndarray:
    """Load a model embedding matrix from disk, optionally L2-normalising rows."""
    return prepare_embedding(np.load(path), dtype, normalize=normalize)


def list_models(emb_dir: Path) -> list[str]:
    """Sorted model names (``*.npy`` stems) in an embeddings directory."""
    return sorted(p.stem for p in emb_dir.glob("*.npy"))


def first_typed_neighbours(distances, start, labels, sites, slides):
    """Nearest SO/OS ranks and distances for a block of consecutive anchor rows.

    Count eligible candidates in (cosine distance, manifest index) order. An
    absent type has rank -1 and distance NaN. Also count SS neighbours before
    the first typed neighbour for the neighbourhood composition diagnostic.
    """
    block = slice(start, start + len(distances))
    rows = np.arange(len(distances))
    indices = np.arange(len(labels))
    same_label = labels[block, None] == labels
    same_site = sites[block, None] == sites
    eligible = slides[block, None] != slides

    def before(distance, index):
        return (distances < distance[:, None]) | (
            (distances == distance[:, None]) & (indices < index[:, None])
        )

    def nearest(matches):
        index = np.where(matches, distances, np.inf).argmin(axis=1)
        has_match = matches[rows, index]
        distance = np.where(has_match, distances[rows, index], np.inf)
        rank = 1 + np.count_nonzero(eligible & before(distance, index), axis=1)
        return index, distance, np.where(has_match, rank, -1)

    so_idx, so_dist, so_rank = nearest(eligible & same_label & ~same_site)
    os_idx, os_dist, os_rank = nearest(eligible & ~same_label & same_site)
    so_first = (so_dist < os_dist) | ((so_dist == os_dist) & (so_idx < os_idx))
    first_idx = np.where(so_first, so_idx, os_idx)
    first_dist = np.minimum(so_dist, os_dist)
    pre_ss = np.count_nonzero(
        eligible & same_label & same_site & before(first_dist, first_idx), axis=1
    )
    return (
        so_rank,
        os_rank,
        np.where(so_rank > 0, so_dist, np.nan),
        np.where(os_rank > 0, os_dist, np.nan),
        pre_ss,
    )
