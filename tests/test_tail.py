from __future__ import annotations

import pytest

from mari import lower_tail_mean, tail_percentile


def test_tail_percentile_and_lower_tail_mean() -> None:
    values = [0.0, 0.5, 1.0]
    assert tail_percentile(values, 50.0) == 0.5
    assert lower_tail_mean(values, 50.0) == 0.25


def test_tail_alpha_validation() -> None:
    with pytest.raises(ValueError):
        tail_percentile([0.1, 0.2], -1.0)
    with pytest.raises(ValueError):
        lower_tail_mean([0.1, 0.2], 101.0)


def test_tail_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        tail_percentile([], 10.0)
    with pytest.raises(ValueError):
        lower_tail_mean([], 10.0)

