"""A benchmark is a row-view over a tileset; identity is the tile, not the label."""

import numpy as np
import pandas as pd
import pytest

from croma.alignment import build_view_row_index


def _tileset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "image_path": ["/img/a.png", "/img/b.png", "/img/c.png", "/img/d.png"],
            "label": ["tumor", "tumor", "benign", "benign"],
            "confounder": ["c1", "c2", "c1", "c2"],
            "slide_id": ["s1", "s2", "s3", "s4"],
        }
    )


def _view(sample_ids: list[str], labels: list[str] | None = None) -> pd.DataFrame:
    paths = [f"/img/{s}.png" for s in sample_ids]
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": paths,
            "label": labels if labels is not None else ["x"] * len(sample_ids),
            "confounder": ["c1"] * len(sample_ids),
            "slide_id": ["s"] * len(sample_ids),
        }
    )


def test_maps_a_subset_to_its_tileset_rows() -> None:
    rows = build_view_row_index(_view(["c", "a"]), _tileset())

    assert rows.tolist() == [2, 0]


def test_relabelling_a_tile_does_not_change_its_identity() -> None:
    # The four-class prostate benchmark calls a tile 'gleason-3' where the binary
    # tileset calls it 'tumor'. Same tile, same embedding row.
    relabelled = _view(["a", "b"], labels=["gleason-3", "gleason-4"])

    rows = build_view_row_index(relabelled, _tileset())

    assert rows.tolist() == [0, 1]


def test_repeated_rows_are_allowed_for_paired_designs() -> None:
    # A paired_2x2 manifest lists a tile once per subset it belongs to.
    rows = build_view_row_index(_view(["a", "b", "a"]), _tileset())

    assert rows.tolist() == [0, 1, 0]


def test_rejects_a_sample_absent_from_the_tileset() -> None:
    with pytest.raises(ValueError, match="not in the tileset"):
        build_view_row_index(_view(["a", "zzz"]), _tileset())


def test_rejects_a_tile_whose_image_path_disagrees() -> None:
    view = _view(["a"])
    view.loc[0, "image_path"] = "/other-root/a.png"

    with pytest.raises(ValueError, match="but the tileset embedded"):
        build_view_row_index(view, _tileset())


def test_one_sample_id_may_name_two_distinct_tiles() -> None:
    # Identity is (sample_id, image_path): a manifest that reuses a sample_id across
    # two different images describes two tiles, not one.
    tileset = _tileset()
    tileset.loc[1, "sample_id"] = "a"  # rows 0 and 1 share sample_id, differ in path
    view = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "image_path": ["/img/b.png", "/img/a.png"],
            "label": ["x", "x"],
            "confounder": ["c1", "c1"],
            "slide_id": ["s", "s"],
        }
    )

    assert build_view_row_index(view, tileset).tolist() == [1, 0]


def test_rejects_a_tileset_with_a_duplicated_tile() -> None:
    tileset = _tileset()
    tileset.loc[1, "sample_id"] = "a"
    tileset.loc[1, "image_path"] = "/img/a.png"

    with pytest.raises(ValueError, match="duplicate tile"):
        build_view_row_index(_view(["a"]), tileset)


def test_rejects_frames_missing_identity_columns() -> None:
    with pytest.raises(ValueError, match="tile-identity columns"):
        build_view_row_index(_view(["a"]).drop(columns=["image_path"]), _tileset())


def test_gathering_features_by_row_index_selects_the_view() -> None:
    features = np.arange(8, dtype=float).reshape(4, 2)

    rows = build_view_row_index(_view(["d", "a"]), _tileset())

    assert features[rows].tolist() == [[6.0, 7.0], [0.0, 1.0]]
