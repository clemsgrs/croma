
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TailMetrics:
    alpha: float
    q_alpha: float
    ltm_alpha: float
    n_tail_samples: int


def compute_tail_metrics(
    sample_values: np.ndarray,
    alpha: float = 0.10,
) -> TailMetrics:
    """Compute threshold-based scalar tail summaries using value <= Q_alpha."""
    arr = np.asarray(sample_values, dtype=float)
    finite = arr[np.isfinite(arr)]

    if len(finite) == 0:
        return TailMetrics(
            alpha=alpha,
            q_alpha=float("nan"),
            ltm_alpha=float("nan"),
            n_tail_samples=0,
        )

    q = float(np.percentile(finite, alpha * 100))
    tail_mask = finite <= q
    tail = finite[tail_mask]

    return TailMetrics(
        alpha=alpha,
        q_alpha=q,
        ltm_alpha=float(tail.mean()) if len(tail) > 0 else q,
        n_tail_samples=int(tail_mask.sum()),
    )


def select_exact_size_tail_set(
    sample_table: pd.DataFrame,
    *,
    value_column: str,
    alpha: float = 0.10,
    sample_id_column: str = "sample_id",
) -> pd.DataFrame:
    """Select exactly ceil(alpha * n_defined) rows, breaking ties by sample_id."""
    if not isinstance(sample_table, pd.DataFrame):
        raise TypeError("sample_table must be a pandas.DataFrame")
    if value_column not in sample_table.columns:
        raise ValueError(f"missing required value column: {value_column}")
    if sample_id_column not in sample_table.columns:
        raise ValueError(f"missing required sample id column: {sample_id_column}")
    if float(alpha) <= 0.0 or float(alpha) > 1.0:
        raise ValueError("alpha must be in (0, 1]")

    ranked = sample_table.copy()
    ranked["_tail_value"] = pd.to_numeric(ranked[value_column], errors="coerce")
    ranked["_tail_sample_id"] = ranked[sample_id_column].astype(str)
    ranked = ranked.loc[np.isfinite(ranked["_tail_value"])].copy()
    if ranked.empty:
        return ranked.drop(columns=["_tail_value", "_tail_sample_id"]).reset_index(drop=True)

    n_tail = int(np.ceil(float(alpha) * float(len(ranked))))
    ranked = ranked.sort_values(
        by=["_tail_value", "_tail_sample_id"],
        ascending=[True, True],
        kind="mergesort",
    )
    selected = ranked.head(n_tail).copy()
    return selected.drop(columns=["_tail_value", "_tail_sample_id"]).reset_index(drop=True)
