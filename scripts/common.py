from __future__ import annotations


def parse_k_candidates(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("k-candidates must include at least one integer")
    return values

