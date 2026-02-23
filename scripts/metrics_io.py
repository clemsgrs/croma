from __future__ import annotations

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


def _cached_row_is_compatible(
    row: pd.Series,
    mode: str,
    tau: float,
    k_candidates_sig: str,
    excluded_centers_sig: str = "",
) -> bool:
    return (
        str(row.get("mode", "")) == str(mode)
        and float(row.get("tau", -1.0)) == float(tau)
        and str(row.get("k_candidates", "")) == str(k_candidates_sig)
        and str(row.get("excluded_centers", "")) == str(excluded_centers_sig)
    )


def load_cached_rows(
    metrics_csv: Path,
    models: list[str],
    mode: str,
    tau: float,
    k_candidates_sig: str,
    excluded_centers_sig: str = "",
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
        compatible_mask = rows.apply(
            lambda row: _cached_row_is_compatible(
                row=row,
                mode=mode,
                tau=tau,
                k_candidates_sig=k_candidates_sig,
                excluded_centers_sig=excluded_centers_sig,
            ),
            axis=1,
        )
        compatible_rows = rows[compatible_mask]
        if compatible_rows.empty:
            continue
        out[model] = compatible_rows.iloc[0].to_dict()
    return out


def save_metrics(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def load_cached_k_sweep_rows(
    metrics_csv: Path,
    models: list[str],
    mode: str,
    tau: float,
    k_candidates_sig: str,
    excluded_centers_sig: str = "",
    expected_k_values: list[int] | tuple[int, ...] | None = None,
) -> dict[str, list[dict]]:
    if not metrics_csv.exists():
        return {}
    df = pd.read_csv(metrics_csv)
    if df.empty or "model" not in df.columns:
        return {}

    out: dict[str, list[dict]] = {}
    expected_k_set = set(int(k) for k in expected_k_values) if expected_k_values is not None else None
    for model in models:
        rows = df[df["model"] == model].copy()
        if rows.empty:
            continue
        compatible_mask = rows.apply(
            lambda row: _cached_row_is_compatible(
                row=row,
                mode=mode,
                tau=tau,
                k_candidates_sig=k_candidates_sig,
                excluded_centers_sig=excluded_centers_sig,
            ),
            axis=1,
        )
        compatible_rows = rows[compatible_mask].copy()
        if compatible_rows.empty:
            continue
        if compatible_rows["k"].duplicated().any():
            continue
        if expected_k_set is not None:
            found_k_set = set(int(k) for k in compatible_rows["k"].tolist())
            if found_k_set != expected_k_set:
                continue
        compatible_rows = compatible_rows.sort_values(by="k", kind="stable")
        out[model] = compatible_rows.to_dict(orient="records")
    return out


def save_k_sweep_metrics(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
