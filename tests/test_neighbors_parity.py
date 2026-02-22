from __future__ import annotations

import logging

import numpy as np
import pytest

from mari.metrics import neighbors as nb


def test_filter_neighbors_excluding_same_slide_does_not_backfill() -> None:
    raw_neighbors = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 0, 1, 3, 4],
            [3, 0, 1, 2, 4],
            [4, 0, 1, 2, 3],
        ],
        dtype=int,
    )
    slide_ids = np.array(["slide-a", "slide-a", "slide-b", "slide-c", "slide-d"], dtype=object)

    neigh, valid_counts = nb._filter_neighbors_excluding_same_slide(
        raw_neighbors=raw_neighbors,
        slide_ids=slide_ids,
        kmax=4,
    )

    assert valid_counts.tolist() == [3, 3, 4, 4, 4]
    assert neigh.shape == (5, 4)
    assert neigh[0].tolist() == [2, 3, 4, -1]
    assert neigh[1].tolist() == [2, 3, 4, -1]

    for sample_idx, row in enumerate(neigh):
        for pos in range(int(valid_counts[sample_idx])):
            assert slide_ids[int(row[pos])] != slide_ids[sample_idx]


def test_predict_labels_uses_per_sample_effective_k() -> None:
    labels = np.array([0, 1, 1, 0, 1], dtype=int)
    neigh = np.array(
        [
            [1, 2, -1],
            [0, 2, 3],
            [0, -1, -1],
            [1, 2, 4],
            [3, 2, -1],
        ],
        dtype=int,
    )
    valid_counts = np.array([2, 3, 1, 3, 2], dtype=int)

    pred, used_mask = nb._predict_labels_from_neighbors(
        labels=labels,
        neigh_idx=neigh,
        valid_counts=valid_counts,
        k=3,
    )

    assert used_mask.tolist() == [True, True, True, True, True]
    assert pred.tolist() == [1, 0, 0, 1, 0]


def test_warn_when_reduced_effective_k_exceeds_ten_percent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="mari")

    nb._warn_if_effective_k_reduced(
        valid_counts=np.array([5] * 9 + [4], dtype=int),
        target_k=5,
        context="toy-dataset",
        threshold=0.10,
    )
    assert not any("effective k < 5" in record.message for record in caplog.records)

    nb._warn_if_effective_k_reduced(
        valid_counts=np.array([5] * 8 + [4, 4], dtype=int),
        target_k=5,
        context="toy-dataset",
        threshold=0.10,
    )
    assert any("effective k < 5" in record.message for record in caplog.records)

