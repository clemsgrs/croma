import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))


def test_first_typed_neighbours_exclude_same_slide_and_use_stable_distance_ties():
    from _neighbor_analysis import first_typed_neighbours

    # Anchor 0. Tile 1 is from its slide, tile 5 is OO. Tied eligible neighbours
    # at distance 0.2 are visited in index order (SO, SS, OS).
    dist = np.array([[0.0, 0.01, 0.2, 0.2, 0.2, 0.1]])
    labels = np.array([0, 1, 0, 0, 1, 1])
    sites = np.array([0, 0, 1, 0, 0, 1])
    slides = np.array([0, 0, 1, 2, 3, 4])
    so, os, so_dist, os_dist, pre_ss = first_typed_neighbours(dist, 0, labels, sites, slides)
    assert (so.tolist(), os.tolist(), pre_ss.tolist()) == ([2], [4], [0])
    assert (so_dist.tolist(), os_dist.tolist()) == ([0.2], [0.2])


def test_first_typed_neighbours_handle_an_absent_type():
    from _neighbor_analysis import first_typed_neighbours

    # Anchor is tile 1; tile 0 is SS and tile 2 is OS.
    dist = np.array([[0.1, 0.0, 0.2]])
    so, os, so_dist, os_dist, pre_ss = first_typed_neighbours(
        dist, 1, np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 1, 2])
    )
    assert (so.tolist(), os.tolist(), pre_ss.tolist()) == ([-1], [2], [1])
    assert np.isnan(so_dist[0])
    assert os_dist.tolist() == [0.2]
