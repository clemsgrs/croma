"""Per-tile colour statistics: what a centre's staining and scanning leave in the pixels.

Two colour descriptions of a tile's tissue, neither of which knows anything about an
encoder:

- **optical density** per RGB channel, ``OD = -log10((I + 1) / 256)``, the quantity stain
  absorbance is linear in (0 for a saturated channel, ``log10 256`` for a black one);
- **CIELAB** ``L*a*b*`` (sRGB, D65 white), a roughly perceptually uniform colour space.

Each is summarised by its mean and population standard deviation over the tile's tissue
pixels: twelve numbers per tile. Tissue is where the optical density summed over the three
channels exceeds ``TISSUE_OD_THRESHOLD``; white glass and background fall below it. No
stain deconvolution: the question is only whether colour alone tells centres apart, not
which stain carries it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: A pixel is tissue when its optical density summed over R, G and B exceeds this. Glass
#: and empty background sit near 0; the faintest eosin is well above it.
TISSUE_OD_THRESHOLD = 0.3

#: The twelve statistics the colour-only centre classifier reads, in a fixed order.
FEATURES = tuple(
    f"{space}_{channel}_{stat}"
    for space, channels in (("od", "rgb"), ("lab", "lab"))
    for channel in channels
    for stat in ("mean", "sd")
)

#: sRGB (linear) to CIE XYZ, and the D65 reference white XYZ is normalised by.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])
_LAB_EPSILON = (6 / 29) ** 3


def srgb_to_lab(rgb) -> np.ndarray:
    """CIELAB ``L*a*b*`` of 8-bit sRGB values (any leading shape, last axis RGB)."""
    c = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T / _D65_WHITE
    f = np.where(xyz > _LAB_EPSILON, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def optical_density(rgb) -> np.ndarray:
    """``-log10((I + 1) / 256)`` per channel of 8-bit RGB values."""
    return -np.log10((np.asarray(rgb, dtype=float) + 1.0) / 256.0)


def colour_statistics(tile) -> dict:
    """The twelve colour statistics of one RGB tile, over its tissue pixels.

    Also returns ``tissue_fraction`` and ``tissue_fallback``: a tile with no pixel above
    ``TISSUE_OD_THRESHOLD`` is summarised over all its pixels instead, and says so.
    """
    pixels = np.asarray(tile)[..., :3].reshape(-1, 3)
    od = optical_density(pixels)
    tissue = od.sum(axis=1) > TISSUE_OD_THRESHOLD
    fallback = not tissue.any()
    chosen = slice(None) if fallback else tissue
    stats = {"tissue_fraction": float(tissue.mean()), "tissue_fallback": bool(fallback)}
    for space, values, channels in (
        ("od", od[chosen], "rgb"),
        ("lab", srgb_to_lab(pixels[chosen]), "lab"),
    ):
        for index, channel in enumerate(channels):
            stats[f"{space}_{channel}_mean"] = float(values[:, index].mean())
            stats[f"{space}_{channel}_sd"] = float(values[:, index].std())
    return stats


#: Bumped whenever ``colour_statistics`` changes what it computes, so a cache written by an
#: older definition is never served.
STATISTICS_VERSION = 1

#: The columns of a cohort colour table, in order.
TABLE_COLUMNS = ("sample_id", "image_path", "image_sha256", "tissue_fraction", "tissue_fallback")


def _cache_key(sample_ids, image_paths) -> str:
    payload = json.dumps(
        {
            "version": STATISTICS_VERSION,
            "tissue_od_threshold": TISSUE_OD_THRESHOLD,
            "features": list(FEATURES),
            "tiles": [[str(s), str(p)] for s, p in zip(sample_ids, image_paths)],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tile_row(item) -> dict:
    """One tile's table row; read once, hashed and decoded from the same bytes."""
    from PIL import Image

    sample_id, image_path = item
    payload = Path(image_path).read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert("RGB"))
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        **colour_statistics(pixels),
    }


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"sample_id": str, "image_path": str, "image_sha256": str},
        keep_default_na=False,
    )


def cohort_colour_stats(sample_ids, image_paths, cache: Path, *, workers: int = 1) -> pd.DataFrame:
    """The colour statistics of every tile, one row each in the given order, cached.

    ``cache`` is a CSV beside a ``.json`` sidecar holding the fingerprint of what it was
    computed from -- the ordered (sample id, image path) list, ``TISSUE_OD_THRESHOLD``,
    ``FEATURES`` and ``STATISTICS_VERSION`` -- and the CSV's own sha256. The cache is served
    only when both match; anything else recomputes it from the images (``workers``
    processes). Each row keeps its image's sha256, so a downstream check can tell whether
    the pixels it summarised are still the ones on disk.
    """
    sample_ids = [str(s) for s in sample_ids]
    image_paths = [str(p) for p in image_paths]
    cache = Path(cache)
    sidecar = cache.with_suffix(".json")
    key = _cache_key(sample_ids, image_paths)
    if cache.exists() and sidecar.exists():
        stored = json.loads(sidecar.read_text())
        if stored.get("key") == key and stored.get("sha256") == _file_sha256(cache):
            return _read_table(cache)
    items = list(zip(sample_ids, image_paths))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(_tile_row, items, chunksize=64))
    else:
        rows = [_tile_row(item) for item in items]
    table = pd.DataFrame(rows, columns=[*TABLE_COLUMNS, *FEATURES])
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, cache)
    sidecar.write_text(
        json.dumps(
            {
                "key": key,
                "sha256": _file_sha256(cache),
                "n_tiles": len(table),
                "tissue_od_threshold": TISSUE_OD_THRESHOLD,
                "statistics_version": STATISTICS_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return _read_table(cache)


#: Folds of the colour-only centre classifier's slide-grouped cross-validation.
CLASSIFIER_FOLDS = 5


@dataclass(frozen=True)
class ColourClassifier:
    """Out-of-fold results of the colour-only centre classifier, one entry per tile.

    ``confidence`` is the out-of-fold probability the classifier gives the tile's *own*
    centre; ``balanced_accuracy`` scores its out-of-fold predicted centres; ``folds`` is the
    fold each tile was scored in.
    """

    confidence: np.ndarray
    balanced_accuracy: float
    folds: np.ndarray


def colour_centre_classifier(
    features, centres, slides, *, n_folds: int = CLASSIFIER_FOLDS, seed: int = 0
) -> ColourClassifier:
    """How well colour statistics alone name a tile's centre, scored on unseen slides.

    A multinomial logistic regression on standardised ``features`` (one row per tile), fitted
    and scored in ``n_folds`` stratified folds that keep every slide in one fold, so no tile
    is scored by a model trained on its own slide's colour.

    Raises:
        ValueError: If a training fold lacks a centre.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=float)
    names, y = np.unique(np.asarray(centres).astype(str), return_inverse=True)
    y = y.reshape(-1)
    probability = np.zeros((len(y), len(names)))
    folds = np.full(len(y), -1)
    splitter = StratifiedGroupKFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    for fold, (train, test) in enumerate(splitter.split(x, y, groups=np.asarray(slides))):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        model.fit(x[train], y[train])
        if len(model.classes_) != len(names):
            raise ValueError(f"fold {fold}: training tiles lack a centre")
        probability[test] = model.predict_proba(x[test])
        folds[test] = fold
    return ColourClassifier(
        confidence=probability[np.arange(len(y)), y],
        balanced_accuracy=float(balanced_accuracy_score(y, probability.argmax(axis=1))),
        folds=folds,
    )
