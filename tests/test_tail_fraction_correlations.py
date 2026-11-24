import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/repro"))

from generate_tail_fraction_table import rank_correlations  # noqa: E402


def test_fraction_changes_the_order_of_the_three_encoders():
    margins = {
        "single low tile": np.array([-1.0] + [0.49, 0.50, 0.51] + [0.8] * 16),
        "four low tiles": np.array([-0.7] * 4 + [0.6] * 16),
        "uniform": np.full(20, 0.8),
    }
    assert tuple(round(r, 3) for r in rank_correlations(margins)) == (0.5, 1.0, 0.5)
