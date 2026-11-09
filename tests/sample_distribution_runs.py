"""A run whose sample distributions sit in the layout but whose rows name a deleted checkout.

Shared by ``test_sample_distribution_paths.py`` (the layout and benchmark, which ships in
the sdist) and ``test_sample_distribution_paper.py`` (the paper's F(0) macros, which does
not).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BENCH = Path(__file__).resolve().parents[1] / "scripts" / "bench"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import layout  # noqa: E402

#: The checkout that ran the benchmark, long since deleted.
GONE = Path("/var/tmp/no-such-checkout")

#: Per-sample CRoMa per model, keyed by registry name. F(0) is the closed-boundary fraction
#: at or below zero: 0, 1/4, 1/2 and 3/4, so it rises with the confounder probe while pooled
#: CRoMa falls. ``Mettle`` is published as "Mettle-1 (Pre-release)", so a reader holding the
#: published frame must map the name back to find the file.
DISTRIBUTIONS = {
    "Alpha": [0.10, 0.20, 0.30, 0.40],
    "Beta": [-0.10, 0.20, 0.30, 0.40],
    "Gamma": [-0.10, 0.00, 0.30, 0.40],
    "Mettle": [-0.30, -0.20, -0.10, 0.40],
}


def stale_run(results_dir: Path) -> list[dict]:
    """A run whose distributions sit in the layout but whose rows name a deleted checkout."""
    rows = []
    for rank, (model, values) in enumerate(DISTRIBUTIONS.items()):
        np.save(_ensure(layout.sample_distribution_path(results_dir, "croma", model)), values)
        stale = GONE / "output" / "metrics" / "median-k" / "toy" / "results"
        rows.append(
            {
                "model": model,
                "croma": 0.4 - 0.1 * rank,
                "ri": 0.3 - 0.1 * rank,
                "mari": 0.35 - 0.1 * rank,
                "croma_ltm_alpha": -0.1 * rank,
                "confounder_knn_bacc": 0.6 + 0.1 * rank,
                "bio_knn_bacc": 0.9 - 0.05 * rank,
                "croma_samples_path": str(layout.sample_distribution_path(stale, "croma", model)),
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "metrics.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(rows), encoding="utf-8")
    return rows


def directory(path: Path) -> Path:
    """``path``, once it exists as a directory."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure(path: Path) -> Path:
    """``path``, once the directory that holds it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
