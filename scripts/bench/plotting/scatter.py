"""Scatter archetype: one-point-per-model scatters.

Every wrapper builds on the shared ``_draw_model_scatter`` primitive in ``base``
via a thin per-comparison ``_draw_*`` helper (biological vs confounder, MaRI vs
RI, CRoMa vs MaRI, Q-alpha vs CRoMa) or, for the CRoMa/LTM scatter, by calling
the primitive directly. The named wrappers stay separate for discoverability.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from croma.plotstyle import COL_ONEHALF

from .base import (
    _confounder_display_name,
    _draw_model_scatter,
    _ltm_label,
    _padded_signed_limits,
    _padded_unit_interval_limits,
    _valid_croma_ltm_rows,
)
from .export import _finalize_figure, _finalize_single_panel_legend_figure


def _draw_bio_vs_confounder_scatter(ax, rows: list[dict]) -> None:
    if not rows:
        return
    confounder_display_name = _confounder_display_name(rows)
    xs = np.asarray([float(r["confounder_knn_bacc"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["bio_knn_bacc"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)
    _draw_model_scatter(
        ax,
        rows,
        x_key="confounder_knn_bacc",
        y_key="bio_knn_bacc",
        xlabel=f"{confounder_display_name} accuracy",
        ylabel="Biological accuracy",
        title=f"Biological vs {confounder_display_name} accuracy",
        xlim=lim,
        ylim=lim,
        diagonal=True,
    )


def _draw_mari_vs_ri_scatter(ax, rows: list[dict]) -> None:
    if not rows:
        return
    xs = np.asarray([float(r["ri"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)
    _draw_model_scatter(
        ax,
        rows,
        x_key="ri",
        y_key="mari",
        xlabel="RI",
        ylabel="MaRI",
        title="MaRI vs RI",
        xlim=lim,
        ylim=lim,
        diagonal=True,
    )


def _draw_croma_vs_mari_scatter(ax, rows: list[dict]) -> None:
    croma_rows = [
        r
        for r in rows
        if "croma" in r and "mari" in r and np.isfinite(float(r["croma"]))
    ]
    if not croma_rows:
        ax.set_visible(False)
        return
    xs = np.asarray([float(r["mari"]) for r in croma_rows], dtype=float)
    ys = np.asarray([float(r["croma"]) for r in croma_rows], dtype=float)
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    _draw_model_scatter(
        ax,
        croma_rows,
        x_key="mari",
        y_key="croma",
        xlabel="MaRI",
        ylabel="CRoMa",
        title="CRoMa vs MaRI",
        xlim=_padded_unit_interval_limits(xs),
        ylim=(float(ys.min()) - y_pad, float(ys.max()) + y_pad),
        hline=0.0,
        vline=0.5,
    )


def _draw_q_alpha_vs_croma_scatter(ax, rows: list[dict]) -> None:
    valid_rows = [
        r
        for r in rows
        if "croma" in r
        and "croma_q_alpha" in r
        and np.isfinite(float(r["croma"]))
        and np.isfinite(float(r["croma_q_alpha"]))
    ]
    if not valid_rows:
        ax.set_visible(False)
        return

    alpha_pct_values = [
        int(round(float(r["croma_alpha"]) * 100))
        for r in valid_rows
        if "croma_alpha" in r and np.isfinite(float(r["croma_alpha"]))
    ]
    alpha_pct = alpha_pct_values[0] if alpha_pct_values else 10

    xs = np.asarray([float(r["croma"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["croma_q_alpha"]) for r in valid_rows], dtype=float)
    x_pad = max(0.1, (xs.max() - xs.min()) * 0.10) if xs.size > 0 else 0.5
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="croma",
        y_key="croma_q_alpha",
        xlabel="CRoMa",
        ylabel=f"Q{alpha_pct}",
        title=f"Q{alpha_pct} vs CRoMa",
        xlim=(float(xs.min()) - x_pad, float(xs.max()) + x_pad),
        ylim=(float(ys.min()) - y_pad, float(ys.max()) + y_pad),
        hline=0.0,
        vline=0.0,
    )


def plot_bio_vs_confounder_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_bio_vs_confounder_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_mari_vs_ri_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_mari_vs_ri_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_croma_ltm_scatter(rows: list[dict], out_path: Path) -> None:
    """CRoMa vs LTM scatter with a horizontal CRoMa=0 robustness threshold.

    The threshold line (not a y=x diagonal) makes the claim non-tautological: every
    model's fragile decile falling below it is an empirical fact, since LTM <= median
    CRoMa by construction would only force points below the diagonal, not below 0.
    """
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    valid_rows = _valid_croma_ltm_rows(rows)

    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    label_ltm = _ltm_label(valid_rows)
    xs = np.asarray([float(r["croma"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ltm"]) for r in valid_rows], dtype=float)
    lim = _padded_signed_limits(np.concatenate([xs, ys]))

    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="croma",
        y_key="ltm",
        xlabel="CRoMa",
        ylabel=label_ltm,
        title=f"CRoMa vs {label_ltm}",
        xlim=lim,
        ylim=lim,
        hline=0.0,
    )

    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_croma_vs_mari_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_croma_vs_mari_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_q_alpha_vs_croma_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_q_alpha_vs_croma_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)
