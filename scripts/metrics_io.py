from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _cached_row_is_compatible(row: pd.Series, mode: str, tau: float, alpha: float) -> bool:
    return (
        str(row.get("mode", "")) == str(mode)
        and float(row.get("tau", -1.0)) == float(tau)
        and float(row.get("alpha", -1.0)) == float(alpha)
    )


def load_cached_rows(
    metrics_csv: Path,
    models: list[str],
    mode: str,
    tau: float,
    alpha: float,
) -> dict[str, dict]:
    if not metrics_csv.exists():
        return {}
    df = pd.read_csv(metrics_csv)
    if df.empty or "model" not in df.columns:
        return {}

    out: dict[str, dict] = {}
    for model in models:
        rows = df[df["model"] == model]
        if rows.empty:
            continue
        row = rows.iloc[0]
        if _cached_row_is_compatible(row=row, mode=mode, tau=tau, alpha=alpha):
            out[model] = row.to_dict()
    return out


def save_metrics(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

