import json
from pathlib import Path

import pandas as pd


def k_candidates_signature(k_candidates: list[int] | tuple[int, ...]) -> str:
    uniq = sorted({int(k) for k in k_candidates})
    if not uniq:
        raise ValueError("k_candidates must include at least one integer")
    if min(uniq) <= 0:
        raise ValueError("k_candidates must be strictly positive")
    return ",".join(str(int(k)) for k in uniq)


def excluded_centers_signature(excluded_centers: list[str] | tuple[str, ...] | None) -> str:
    if not excluded_centers:
        return ""
    uniq = sorted({str(center).strip() for center in excluded_centers if str(center).strip()})
    return ",".join(uniq)


def ccrr_search_signature(
    *,
    start_k: int,
    k_growth_factor: float,
    alpha: float,
) -> str:
    return (
        f"start={int(start_k)};"
        f"growth={float(k_growth_factor):.8g};"
        f"alpha={float(alpha):.8g}"
    )


def save_metrics(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


