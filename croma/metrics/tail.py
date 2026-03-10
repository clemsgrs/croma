
from dataclasses import dataclass

import numpy as np


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
