"""Harness for the metric property suite: named embeddings plus a metric adapter.

Two things live here, and nothing else:

*Named embeddings.* Small synthetic embeddings whose geometry is known by construction,
each a constructor returning ``(features, manifest)``. The names come from the glossary in
``CONTEXT.md`` -- *biology-dominant*, *confounder-dominant*, *exactly contested* -- so a
test reads as a claim about a neighbourhood regime rather than about an array.

*A metric adapter.* ``RI``/``MaRI`` take ``k_candidates`` and return a ``RobustnessResult``;
``CRoMa`` takes ``m``/``start_k``/``k_growth_factor`` and returns a ``CRoMaResult``.
:func:`compute_metric` hides that signature difference and nothing else, so a shared
property can parametrize over ``METRICS`` instead of being written once per metric.

To add a named embedding: write a constructor returning ``(features, manifest)`` -- build
the manifest with :func:`toy_manifest` so the required columns and the one-tile-per-slide
convention come for free -- and register it in :data:`NAMED_EMBEDDINGS`. Keep every
candidate ``k`` strictly below the number of rows in the neighbourhood scope (see
:data:`PINNED_K`), and keep each of the four label-confounder cells at least
:data:`CELL_DEPTH` rows deep so CRoMa is defined at the default ``m``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from croma import CRoMa, MaRI, RI
from croma.metrics.croma import CROMA_HEADLINE_M

#: The confounder column the named embeddings carry. Not the canonical name on purpose:
#: ``compute`` has to normalize it, which is part of the chain under test.
CONFOUNDER_COLUMN = "scanner_vendor"

#: Named embeddings are scored dataset-wide, so the neighbourhood scope is the whole
#: manifest and the k-versus-scope-size constraint reads off ``len(manifest)`` directly.
EVALUATION_DESIGN = "dataset_wide"

LABELS = ("A", "B")
CONFOUNDERS = ("V1", "V2")

#: The single pinned ``k``. Every named embedding has strictly more rows than this in its
#: neighbourhood scope -- below that, ``compute`` raises rather than scoring on a truncated
#: neighbourhood. It is also small enough to stay inside a cluster in the dominance
#: embeddings, which is what makes their neighbour types knowable by construction.
PINNED_K = 3

#: CRoMa's per-type averaging radius. The library headline, so the property tests exercise
#: the configuration the paper reports; every named embedding is sized to define it.
DEFAULT_M = CROMA_HEADLINE_M

#: Rows per label-confounder cell in every named embedding. One more than ``DEFAULT_M``, not
#: exactly ``DEFAULT_M``: the neighbour search caps its fetch at ``n_samples - 1`` *including
#: the sample itself*, so the farthest candidate is never visible. With cells exactly ``m``
#: deep that costs some samples their ``m``-th typed neighbour and CRoMa goes undefined on
#: them; one spare row per cell keeps every sample defined.
CELL_DEPTH = DEFAULT_M + 1

#: The metrics a shared property parametrizes over. Adding a fourth metric means adding it
#: here and teaching :func:`compute_metric` its signature.
METRICS = (RI, MaRI, CRoMa)


def metric_id(metric: type) -> str:
    """Readable ``pytest.mark.parametrize`` id for a metric class."""
    return {RI: "RI", MaRI: "MaRI", CRoMa: "CRoMa"}[metric]


@dataclass(frozen=True)
class MetricScore:
    """The two things every metric returns, under one name.

    ``score`` is the pooled scalar (``result.value``) and ``sample_values`` the per-sample
    values on which the metric is defined. Their meaning differs per metric -- RI/MaRI are
    SO-versus-OS ratios in ``[0, 1]``, CRoMa a signed margin in ``(-1, 1)`` -- and the
    adapter deliberately does not reconcile that.
    """

    score: float
    sample_values: np.ndarray


def compute_metric(
    metric: type,
    features: np.ndarray,
    manifest: pd.DataFrame,
    *,
    k: int = PINNED_K,
    m: int = DEFAULT_M,
    confounder_column: str = CONFOUNDER_COLUMN,
    evaluation_design: str = EVALUATION_DESIGN,
    **kwargs,
) -> MetricScore:
    """Compute ``metric`` at a pinned ``k`` and normalize the result to a ``MetricScore``.

    ``k`` pins RI/MaRI to a single candidate, which removes k-selection from the picture
    without weakening the property, since the invariants hold at every fixed ``k``. CRoMa
    has no k-selection to pin: it grows its radius until each sample has ``m`` neighbours of
    each type, so ``k`` sets ``start_k`` -- the radius the search starts from.

    Extra keyword arguments (``tau``, ``alpha``, ...) pass through to the metric untouched.
    """
    if metric is CRoMa:
        result = metric.compute(
            features,
            manifest,
            confounder_column=confounder_column,
            evaluation_design=evaluation_design,
            m=int(m),
            start_k=int(k),
            **kwargs,
        )
    else:
        result = metric.compute(
            features,
            manifest,
            confounder_column=confounder_column,
            evaluation_design=evaluation_design,
            k_candidates=[int(k)],
            **kwargs,
        )
    return MetricScore(
        score=float(result.value),
        sample_values=np.asarray(result.sample_values, dtype=float),
    )


def orthogonal_matrix(dim: int, *, seed: int = 20260727) -> np.ndarray:
    """A seeded orthogonal ``(dim, dim)`` matrix, so rotation tests are deterministic.

    QR of a Gaussian matrix, with the sign convention fixed by the diagonal of ``R`` -- QR
    is only unique up to those signs, and pinning them makes the matrix a function of the
    seed alone rather than of the LAPACK build.
    """
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((int(dim), int(dim))))
    return q * np.sign(np.diag(r))


def toy_manifest(labels: list[str], confounders: list[str]) -> pd.DataFrame:
    """Manifest for ``len(labels)`` samples, one tile per slide.

    Distinct ``slide_id`` per row on purpose: the same-slide exclusion then removes nothing,
    so the neighbour sets are exactly the ones the geometry implies.
    """
    n = len(labels)
    if len(confounders) != n:
        raise ValueError("labels and confounders must have the same length")
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tmp/{i}.png" for i in range(n)],
            "label": list(labels),
            CONFOUNDER_COLUMN: list(confounders),
            "slide_id": [f"slide-{i}" for i in range(n)],
            "dataset": ["toy"] * n,
        }
    )


def biology_dominant(
    *, n_per_cell: int = CELL_DEPTH, spread: float = 0.01
) -> tuple[np.ndarray, pd.DataFrame]:
    """Label clusters tight, confounder scattered inside them.

    Each label owns its own basis direction, so cross-label cosine distance is ~1 while
    within-cluster distance is ~1e-4: every top-``PINNED_K`` neighbour is same-label. The
    two confounder values alternate along the within-cluster axis, so those neighbours are a
    mix of SS and SO and never OS -- ``os_total == 0`` and pooled RI is exactly 1.

    Each label-confounder cell holds ``n_per_cell`` rows, so CRoMa finds its ``m`` SO and
    ``m`` OS neighbours once the search radius grows past the label cluster.
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    confounders: list[str] = []
    for label_idx, label in enumerate(LABELS):
        for i in range(2 * n_per_cell):
            vec = np.zeros(len(LABELS) + 1, dtype=float)
            vec[label_idx] = 1.0
            vec[-1] = spread * i
            rows.append(vec)
            labels.append(label)
            confounders.append(CONFOUNDERS[i % len(CONFOUNDERS)])
    return np.asarray(rows, dtype=float), toy_manifest(labels, confounders)


def confounder_dominant(
    *, n_per_cell: int = CELL_DEPTH, spread: float = 0.01
) -> tuple[np.ndarray, pd.DataFrame]:
    """The mirror of :func:`biology_dominant`: confounder clusters tight, labels scattered.

    Each confounder value owns a basis direction, so every top-``PINNED_K`` neighbour is
    same-confounder. The two labels alternate along the within-cluster axis, so those
    neighbours are a mix of SS and OS and never SO -- ``so_total == 0`` and pooled RI is
    exactly 0.
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    confounders: list[str] = []
    for confounder_idx, confounder in enumerate(CONFOUNDERS):
        for i in range(2 * n_per_cell):
            vec = np.zeros(len(CONFOUNDERS) + 1, dtype=float)
            vec[confounder_idx] = 1.0
            vec[-1] = spread * i
            rows.append(vec)
            labels.append(LABELS[i % len(LABELS)])
            confounders.append(confounder)
    return np.asarray(rows, dtype=float), toy_manifest(labels, confounders)


def contested(
    *, n_blocks: int = CELL_DEPTH, spread: float = 0.1
) -> tuple[np.ndarray, pd.DataFrame]:
    """Exactly contested: one SO and one OS equidistant from every sample.

    ``n_blocks`` blocks, each a 2x2 square of the four label-confounder cells laid out in a
    shared plane around its own basis direction. Within a block the SO and the OS neighbour
    sit on perpendicular corners and the OO neighbour on the opposite one, so each sample's
    nearest SO and nearest OS are the same distance away and its metric value is the
    boundary: RI and MaRI 0.5, CRoMa 0. Across blocks the SO and OS corners are equidistant
    too, so the symmetry survives up to ``m = n_blocks``.

    WARNING -- this embedding contains **exact distance ties by construction**: the SO and
    OS neighbours are equidistant to the bit, so their kNN ordering is arbitrary and depends
    on the tie-break inside the neighbour search. Every metric here is invariant to that
    ordering (both are inside the top-k, and their distances are equal), but a
    permutation-invariance test is not: permuting the rows can reorder tied neighbours and
    change which one is picked. Use a tie-free embedding for permutation testing.
    """
    # Square corners, in the order (label, confounder) = (A,V1), (A,V2), (B,V1), (B,V2).
    # Adjacent corners are perpendicular and opposite corners antipodal, which is what makes
    # a sample's SO and OS neighbours equidistant and its OO neighbour strictly farther.
    corners = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]], dtype=float)
    cells = [(label, confounder) for label in LABELS for confounder in CONFOUNDERS]

    dim = n_blocks + corners.shape[1]
    rows: list[np.ndarray] = []
    labels: list[str] = []
    confounders: list[str] = []
    for block in range(n_blocks):
        for corner, (label, confounder) in zip(corners, cells):
            vec = np.zeros(dim, dtype=float)
            vec[block] = 1.0
            vec[n_blocks:] = spread * corner
            rows.append(vec)
            labels.append(label)
            confounders.append(confounder)
    return np.asarray(rows, dtype=float), toy_manifest(labels, confounders)


#: Every named embedding, by name. Shared properties parametrize over this; add a new
#: constructor here and it is picked up by all of them.
NAMED_EMBEDDINGS = {
    "biology_dominant": biology_dominant,
    "confounder_dominant": confounder_dominant,
    "contested": contested,
}


def _cycled_cells(n_per_cell: int) -> tuple[list[str], list[str]]:
    """Label/confounder assignments cycling through the four cells, ``n_per_cell`` times each.

    The cycle is fixed rather than sampled, so the four label-confounder cells are exactly
    balanced. For an embedding whose geometry carries no label or confounder structure that
    balance is what makes the null anchor exact: SO and OS are equally available a priori.
    """
    labels: list[str] = []
    confounders: list[str] = []
    for _ in range(int(n_per_cell)):
        for label in LABELS:
            for confounder in CONFOUNDERS:
                labels.append(label)
                confounders.append(confounder)
    return labels, confounders


def constant_embedding(
    *, n_per_cell: int = CELL_DEPTH, dim: int = 8
) -> tuple[np.ndarray, pd.DataFrame]:
    """Every row the same vector, so every cosine distance is exactly zero.

    The degenerate embedding: an encoder that has collapsed. Distances carry no information,
    which is a different thing from carrying neutral information, and the metrics part ways
    here. CRoMa's ``(d_OS - d_SO) / (d_OS + d_SO)`` has a zero denominator and is undefined;
    RI and MaRI still count neighbours (MaRI's weights are ``exp(-0 / tau) = 1``) and so
    degrade to a count ratio over an arbitrary tie-break rather than to nothing.

    Deliberately **not** in :data:`NAMED_EMBEDDINGS`: the shared properties there compare
    scores with ``pytest.approx``, under which NaN is unequal to NaN, so a metric that is
    undefined everywhere would turn a rotation-invariance failure into a NaN-comparison
    artifact. Tests that want this embedding ask for it by name.
    """
    labels, confounders = _cycled_cells(n_per_cell)
    features = np.ones((len(labels), int(dim)), dtype=float)
    return features, toy_manifest(labels, confounders)


def isotropic_gaussian(
    *, n_per_cell: int = 250, dim: int = 32, seed: int = 20260727
) -> tuple[np.ndarray, pd.DataFrame]:
    """Seeded iid Gaussian rows, labelled and confounded independently of the geometry.

    The null: neither the label nor the confounder is encoded, so neighbourhoods are typed at
    chance and the metrics have nothing to say. Cells are assigned by a fixed cycle over rows
    the features know nothing about, which is what makes the independence exact rather than
    approximate. ``n_per_cell`` is large so the sampling error around the null stays small
    enough to be asserted tightly.

    Not in :data:`NAMED_EMBEDDINGS` either: its geometry is known only in distribution, while
    that registry is documented as embeddings whose neighbour types are known by
    construction, and the properties parametrized over it are exact.
    """
    labels, confounders = _cycled_cells(n_per_cell)
    rng = np.random.default_rng(int(seed))
    features = rng.standard_normal((len(labels), int(dim)))
    return features, toy_manifest(labels, confounders)
