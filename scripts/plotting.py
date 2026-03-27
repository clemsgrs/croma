import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import gaussian_kde

from croma.confounders import infer_confounder_display_name

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
    }
)

TEXT_COLOR = "#18212b"
GRID_COLOR = "#d8dee7"
SPINE_COLOR = "#c6ced8"
REFERENCE_LINE_COLOR = "#6b7280"
PANEL_FACE_COLOR = "#fdfdfc"
FRAGILE_SHADE_COLOR = "#f7e7d7"
DEFAULT_DPI = 300
LEGEND_Y = 0.02
LEGEND_MAX_COLUMNS = 4

MODEL_COLOR_MAP: dict[str, str] = {
    "Virchow2": "#ff7f0e",
    "Virchow": "#ffbb78",
    "PRISM": "#ff7f0e",
    "UNI2-h": "#2ca02c",
    "UNI": "#98df8a",
    "CONCHv1.5": "#d62728",
    "CONCH": "#ff9896",
    "TITAN": "#d62728",
    "Phikon-v2": "#9467bd",
    "Phikon": "#c5b0d5",
    "H-optimus-1": "#8c564b",
    "H-optimus-0": "#c49c94",
    "H0-mini": "#d7b5b0",
    "Prov-GigaPath": "#1f77b4",
    "Midnight-12k": "#17becf",
    "Hibou-L": "#e377c2",
    "Hibou-B": "#f7b6d2",
    "Prost40M": "#636363",
}

MODEL_FAMILY_MAP: dict[str, str] = {
    "Virchow2": "paige",
    "Virchow": "paige",
    "PRISM": "paige",
    "UNI2-h": "uni",
    "UNI": "uni",
    "CONCHv1.5": "conch",
    "CONCH": "conch",
    "TITAN": "conch",
    "Phikon-v2": "phikon",
    "Phikon": "phikon",
    "H-optimus-1": "hoptimus",
    "H-optimus-0": "hoptimus",
    "H0-mini": "hoptimus",
    "Prov-GigaPath": "gigapath",
    "Midnight-12k": "midnight",
    "Hibou-L": "hibou",
    "Hibou-B": "hibou",
    "Prost40M": "prost",
}

FAMILY_PALETTE: dict[str, list[str]] = {
    "paige": ["#996127", "#cf8f45"],
    "uni": ["#3f7f62", "#76a888"],
    "conch": ["#9e4d4d", "#ce7e74"],
    "phikon": ["#6274a8", "#8e9bc4"],
    "hoptimus": ["#557985", "#809aa3", "#b1bcc2"],
    "gigapath": ["#4d7296"],
    "midnight": ["#3e8c9b"],
    "hibou": ["#8d678b", "#b996b4"],
    "prost": ["#6d6a68"],
    "other": ["#6f7b87", "#98a1ab", "#c0c6cd"],
}

MODEL_TONE_INDEX: dict[str, int] = {
    "Virchow": 0,
    "Virchow2": 1,
    "PRISM": 1,
    "UNI": 0,
    "UNI2-h": 1,
    "CONCH": 0,
    "CONCHv1.5": 1,
    "TITAN": 1,
    "Phikon": 0,
    "Phikon-v2": 1,
    "H-optimus-1": 0,
    "H-optimus-0": 1,
    "H0-mini": 2,
    "Prov-GigaPath": 0,
    "Midnight-12k": 0,
    "Hibou-L": 0,
    "Hibou-B": 1,
    "Prost40M": 0,
}

_SUPPORT_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "good": ("#4f8f76", "#dcebe4"),
    "warning": ("#b88a2a", "#f3e6c8"),
    "critical": ("#c9625b", "#f2d8d5"),
}


def _family_for_model(model: str) -> str:
    return str(MODEL_FAMILY_MAP.get(str(model), "other"))


def _color_for_model(model: str) -> str:
    if str(model) in MODEL_COLOR_MAP:
        return str(MODEL_COLOR_MAP[str(model)])
    family = _family_for_model(model)
    palette = FAMILY_PALETTE.get(family, FAMILY_PALETTE["other"])
    tone_idx = int(MODEL_TONE_INDEX.get(str(model), 0)) % len(palette)
    return str(palette[tone_idx])



def _style_axes(ax, *, grid_axis: str = "both") -> None:
    ax.set_facecolor(PANEL_FACE_COLOR)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(SPINE_COLOR)
        ax.spines[spine_name].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=9.5, width=0.8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    if grid_axis == "y":
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    else:
        ax.grid(color=GRID_COLOR, linewidth=0.8, alpha=0.9)


def _set_panel_title(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, weight="semibold", pad=8)


def _legend_columns(n_labels: int) -> int:
    if n_labels <= 1:
        return 1
    return min(LEGEND_MAX_COLUMNS, max(2, n_labels))


def _collect_legend_entries(axes: list[plt.Axes]) -> tuple[list, list[str]]:
    handles: list = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        if not ax.get_visible():
            continue
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            text = str(label).strip()
            if not text or text.startswith("_") or text in seen:
                continue
            seen.add(text)
            handles.append(handle)
            labels.append(text)
    return handles, labels


def _add_figure_legend(
    fig,
    axes: list[plt.Axes],
    *,
    y: float = LEGEND_Y,
    ncol: int | None = None,
    fontsize: float = 9.0,
    columnspacing: float = 1.4,
    handlelength: float = 2.2,
) -> bool:
    handles, labels = _collect_legend_entries(axes)
    if not handles:
        return False
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=int(ncol) if ncol is not None else _legend_columns(len(labels)),
        frameon=False,
        fontsize=fontsize,
        columnspacing=columnspacing,
        handlelength=handlelength,
        handletextpad=0.6,
    )
    return True


def _legend_bottom_margin(axes: list[plt.Axes]) -> float:
    _handles, labels = _collect_legend_entries(axes)
    if not labels:
        return 0.12
    ncols = _legend_columns(len(labels))
    nrows = int(math.ceil(len(labels) / float(ncols)))
    return min(0.095 + 0.042 * nrows, 0.255)


def _png_export_path(out_path: Path) -> Path:
    return out_path.parent / "png" / out_path.name


def _pdf_export_path(out_path: Path) -> Path:
    return out_path.parent / "pdf" / out_path.with_suffix(".pdf").name


def _finalize_figure(
    fig,
    *,
    out_path: Path,
    legend_axes: list[plt.Axes] | None = None,
    hspace: float = 0.30,
    wspace: float = 0.24,
    add_legend: bool = True,
    left: float = 0.10,
    right: float = 0.98,
    top: float = 0.92,
    bottom: float | None = None,
    legend_y: float = LEGEND_Y,
    legend_ncol: int | None = None,
    legend_fontsize: float = 9.0,
    legend_columnspacing: float = 1.4,
    legend_handlelength: float = 2.2,
) -> None:
    axes = [ax for ax in (legend_axes or list(fig.axes)) if ax.get_visible()]
    bottom_margin = (
        float(bottom)
        if bottom is not None
        else (_legend_bottom_margin(axes) if add_legend else 0.10)
    )
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom_margin,
        hspace=hspace,
        wspace=wspace,
    )
    if add_legend:
        _add_figure_legend(
            fig,
            axes,
            y=legend_y,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            columnspacing=legend_columnspacing,
            handlelength=legend_handlelength,
        )
    png_path = _png_export_path(out_path)
    pdf_path = _pdf_export_path(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DEFAULT_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)


def _finalize_single_panel_legend_figure(fig, *, out_path: Path, ax: plt.Axes) -> None:
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax],
        top=0.94,
        bottom=0.245,
        legend_y=0.016,
        legend_fontsize=8.8,
        legend_columnspacing=1.25,
        legend_handlelength=2.0,
    )


def _clamp_fraction(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _support_status(defined_frac: float) -> str:
    frac = _clamp_fraction(defined_frac)
    if frac < 0.25:
        return "critical"
    if frac < 0.50:
        return "warning"
    return "good"


def _support_plot_rows(rows: list[dict]) -> list[dict]:
    support_rows: list[dict] = []

    for raw_row in rows:
        model = str(raw_row.get("model", "")).strip()
        if not model:
            continue

        undefined_values: list[float] = []
        for key in ("ri_undefined_frac", "mari_undefined_frac"):
            if key not in raw_row:
                continue
            try:
                undefined_frac = float(raw_row[key])
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(undefined_frac):
                continue
            undefined_values.append(_clamp_fraction(undefined_frac))

        if not undefined_values:
            continue

        undefined_frac = float(max(undefined_values))
        defined_frac = float(1.0 - undefined_frac)
        status = _support_status(defined_frac)
        fill_color, track_color = _SUPPORT_STATUS_COLORS[status]
        support_rows.append(
            {
                "model": model,
                "undefined_frac": undefined_frac,
                "defined_frac": defined_frac,
                "status": status,
                "fill_color": fill_color,
                "track_color": track_color,
                "label": f"{int(round(defined_frac * 100.0))}%",
            }
        )

    return sorted(support_rows, key=lambda row: (row["defined_frac"], row["model"]))


def _padded_unit_interval_limits(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = float(vmax - vmin)

    if span <= 1e-9:
        base = max(0.02, abs(vmin) * 0.10)
        lo = max(0.0, vmin - base)
        hi = min(1.0, vmax + base)
        if hi <= lo:
            lo, hi = max(0.0, vmin - 0.05), min(1.0, vmin + 0.05)
        return float(lo), float(hi)

    pad = max(0.02, span * 0.10)
    lo = max(0.0, vmin - pad)
    hi = min(1.0, vmax + pad)
    if hi - lo < 0.05:
        mid = 0.5 * (lo + hi)
        lo = max(0.0, mid - 0.025)
        hi = min(1.0, mid + 0.025)
    return float(lo), float(hi)


def _group_k_rows(rows: list[dict]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["k"]))
    return by_model


def _confounder_display_name(rows: list[dict]) -> str:
    if not rows:
        return "Confounder"
    for row in rows:
        display = str(row.get("confounder_display_name", "")).strip()
        if display:
            return display
        column = str(row.get("confounder_column", "")).strip()
        if column:
            return infer_confounder_display_name(column)
    return "Confounder"


def _set_k_axis(ax, k_ticks: list[int]) -> None:
    _style_axes(ax)
    ax.set_xlabel("k", fontsize=10.5)
    if len(k_ticks) <= 12:
        ax.set_xticks(k_ticks)
    else:
        ax.set_xticks(_human_friendly_integer_ticks(k_ticks))

    if len(k_ticks) > 1:
        span = float(max(k_ticks) - min(k_ticks))
        pad = max(0.5, 0.03 * span)
        ax.set_xlim(float(min(k_ticks)) - pad, float(max(k_ticks)) + pad)
    else:
        ax.set_xlim(float(k_ticks[0]) - 0.5, float(k_ticks[0]) + 0.5)


def _human_friendly_integer_ticks(k_ticks: list[int], *, max_ticks: int = 8) -> list[int]:
    if not k_ticks:
        return []
    k_min = int(min(k_ticks))
    k_max = int(max(k_ticks))
    if len(k_ticks) <= max_ticks and len(k_ticks) == (k_max - k_min + 1):
        return list(k_ticks)

    if k_max <= k_min:
        return [k_min]

    span = max(1, k_max - k_min)
    target_step = max(1.0, span / max(1, max_ticks - 1))
    magnitude = 10 ** math.floor(math.log10(target_step))
    normalized = target_step / magnitude
    if normalized <= 1.0:
        step = 1 * magnitude
    elif normalized <= 2.0:
        step = 2 * magnitude
    elif normalized <= 5.0:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    step = int(max(1, step))

    ticks = [k_min]
    tick = int(math.ceil((k_min + 1) / step) * step)
    while tick < k_max:
        ticks.append(int(tick))
        tick += step
    if ticks[-1] != k_max:
        ticks.append(k_max)
    return ticks


def _highlight_selected(
    ax, *, ks: np.ndarray, ys: np.ndarray, selected_k: int, color: str, marker: str
) -> None:
    idx_arr = np.where(ks == int(selected_k))[0]
    if int(idx_arr.size) == 0:
        return
    idx = int(idx_arr[0])
    size = 220 if marker == "*" else 190
    ax.scatter(
        [ks[idx]],
        [ys[idx]],
        s=size,
        color=color,
        marker=marker,
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )


def _draw_k_curve(
    ax,
    *,
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
    highlight_rules: list[tuple[str, str]],
) -> None:
    if not rows:
        return
    by_model = _group_k_rows(rows)
    models = sorted(by_model)
    k_ticks = sorted({int(row["k"]) for row in rows})
    all_values = np.asarray([float(row[value_key]) for row in rows], dtype=float)

    _set_k_axis(ax, k_ticks)
    ax.set_ylim(*_padded_unit_interval_limits(all_values))
    ax.set_ylabel(ylabel, fontsize=10.5)
    _set_panel_title(ax, title)

    for model in models:
        model_rows = by_model[model]
        ks = np.asarray([int(r["k"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r[value_key]) for r in model_rows], dtype=float)
        color = _color_for_model(model)
        ax.plot(
            ks,
            vals,
            color=color,
            linewidth=1.8,
            alpha=0.95,
            marker=None,
            label=model,
        )
        for selected_key, marker in highlight_rules:
            selected_k = int(
                model_rows[0].get(
                    selected_key, model_rows[0].get("selected_k", model_rows[0]["k"])
                )
            )
            _highlight_selected(
                ax,
                ks=ks,
                ys=vals,
                selected_k=selected_k,
                color=color,
                marker=marker,
            )


def _draw_bio_vs_confounder_scatter(
    ax, rows: list[dict]
) -> None:
    if not rows:
        return

    confounder_display_name = _confounder_display_name(rows)
    xs = np.asarray([float(r["confounder_knn_bacc"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["bio_knn_bacc"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)

    _style_axes(ax)
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.1,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
    )

    for row in sorted(rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["confounder_knn_bacc"])
        y = float(row["bio_knn_bacc"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"{confounder_display_name} Accuracy", fontsize=10.5)
    ax.set_ylabel("Biological Accuracy", fontsize=10.5)
    _set_panel_title(ax, f"Biological vs {confounder_display_name} Accuracy")


def _draw_mari_vs_ri_scatter(
    ax, rows: list[dict]
) -> None:
    if not rows:
        return

    xs = np.asarray([float(r["ri"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)

    _style_axes(ax)
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.1,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
    )

    for row in sorted(rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["ri"])
        y = float(row["mari"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("RI", fontsize=10.5)
    ax.set_ylabel("MaRI", fontsize=10.5)
    _set_panel_title(ax, "MaRI vs RI")


def plot_knn_bio_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_bacc",
        ylabel="Balanced accuracy",
        title="Biological Accuracy over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_knn_confounder_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    confounder_display_name = _confounder_display_name(rows)
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_confounder_bacc",
        ylabel="Balanced accuracy",
        title=f"{confounder_display_name} Accuracy over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_ri_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="ri",
        ylabel="RI",
        title="RI over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_mari_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="mari",
        ylabel="MaRI",
        title="MaRI over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_ri_mari_support(rows: list[dict], out_path: Path) -> None:
    support_rows = _support_plot_rows(rows)
    fig_height = max(3.4, 0.85 + 0.58 * max(len(support_rows), 1))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))

    if not support_rows:
        ax.set_visible(False)
        png_path = _png_export_path(out_path)
        pdf_path = _pdf_export_path(out_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(png_path, dpi=DEFAULT_DPI)
        fig.savefig(pdf_path)
        plt.close(fig)
        return

    y = np.arange(len(support_rows), dtype=float)
    labels = [str(row["model"]) for row in support_rows]

    ax.set_facecolor("#fbfcfd")
    fig.patch.set_facecolor("#fbfcfd")
    ax.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.9, zorder=0)

    for idx, row in enumerate(support_rows):
        defined_frac = float(row["defined_frac"])
        undefined_frac = float(row["undefined_frac"])
        fill_color = str(row["fill_color"])
        track_color = str(row["track_color"])

        ax.barh(
            y[idx],
            1.0,
            color=track_color,
            edgecolor="none",
            height=0.58,
            zorder=1,
        )
        ax.barh(
            y[idx],
            defined_frac,
            color=fill_color,
            edgecolor="none",
            height=0.58,
            zorder=2,
        )

        ax.text(
            defined_frac / 2.0,
            y[idx],
            str(row["label"]),
            va="center",
            ha="center",
            fontsize=10,
            color="white",
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of evaluated samples", fontsize=11, labelpad=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Support Coverage", fontsize=13, weight="bold", pad=6)
    ax.set_ylim(float(len(support_rows) - 0.35), -0.65)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Patch(facecolor=_SUPPORT_STATUS_COLORS["critical"][0], label="Defined <25%"),
        Patch(facecolor=_SUPPORT_STATUS_COLORS["warning"][0], label="Defined <50%"),
        Patch(facecolor=_SUPPORT_STATUS_COLORS["good"][0], label="Defined >=50%"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        ncol=3,
        fontsize=9.5,
    )

    png_path = _png_export_path(out_path)
    pdf_path = _pdf_export_path(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.985))
    fig.savefig(png_path, dpi=DEFAULT_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_ccmr_m_sweep_with_ltm(rows: list[dict], out_path: Path) -> None:
    fig, (ax_ccmr, ax_ltm) = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    ccmr_rows = [
        r
        for r in rows
        if "m" in r
        and "ccmr" in r
        and np.isfinite(float(r["m"]))
        and np.isfinite(float(r["ccmr"]))
    ]
    if not ccmr_rows:
        for ax in (ax_ccmr, ax_ltm):
            ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, add_legend=False)
        return

    by_model: dict[str, list[dict]] = {}
    for row in ccmr_rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["m"]))

    m_all = sorted({int(row["m"]) for row in ccmr_rows})
    m_min, m_max = m_all[0], m_all[-1]

    def _configure_ax(
        ax: plt.Axes, ylabel: str, title: str, values: np.ndarray
    ) -> None:
        _style_axes(ax)
        ax.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1.1,
            color=REFERENCE_LINE_COLOR,
            zorder=1,
            alpha=0.8,
        )
        ax.set_ylabel(ylabel, fontsize=10.5)
        _set_panel_title(ax, title)
        finite = values[np.isfinite(values)]
        vmin = float(np.nanmin(finite)) if finite.size > 0 else 0.0
        vmax = float(np.nanmax(finite)) if finite.size > 0 else 1.0
        span = vmax - vmin
        pad = max(0.05, span * 0.10) if span > 1e-9 else max(0.1, abs(vmin) * 0.10)
        ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    ccmr_values = np.asarray([float(r["ccmr"]) for r in ccmr_rows], dtype=float)
    ltm_values = np.asarray(
        [float(r["ccmr_ltm_alpha"]) for r in ccmr_rows if "ccmr_ltm_alpha" in r],
        dtype=float,
    )
    _configure_ax(ax_ccmr, "CCMR", "CCMR over m", ccmr_values)
    if ltm_values.size > 0:
        _configure_ax(ax_ltm, "LTM", "LTM over m", ltm_values)

    for model in sorted(by_model):
        model_rows = by_model[model]
        color = _color_for_model(model)
        ms = np.asarray([int(r["m"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r["ccmr"]) for r in model_rows], dtype=float)
        ax_ccmr.plot(ms, vals, color=color, linewidth=1.8, alpha=0.95, label=model)
        if "ccmr_ltm_alpha" in model_rows[0]:
            ltms = np.asarray(
                [float(r["ccmr_ltm_alpha"]) for r in model_rows], dtype=float
            )
            ax_ltm.plot(ms, ltms, color=color, linewidth=1.8, alpha=0.95, label=model)

    tick_positions = _human_friendly_integer_ticks(m_all, max_ticks=6)
    ax_ltm.set_xticks(tick_positions)
    ax_ltm.set_xlim(m_min - 0.5, m_max + 0.5)
    ax_ltm.set_xlabel("m", fontsize=10.5)

    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax_ccmr, ax_ltm],
        hspace=0.26,
        bottom=0.18,
        legend_y=0.012,
        legend_ncol=4,
        legend_fontsize=8.4,
        legend_columnspacing=1.1,
        legend_handlelength=1.8,
    )


def _valid_ccmr_ltm_rows(rows: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for row in rows:
        if "ccmr" not in row or "ccmr_ltm_alpha" not in row:
            continue
        try:
            ccmr_value = float(row["ccmr"])
            ltm_value = float(row["ccmr_ltm_alpha"])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(ccmr_value) or not np.isfinite(ltm_value):
            continue

        try:
            alpha_value = float(row.get("ccmr_alpha", float("nan")))
        except Exception:  # noqa: BLE001
            alpha_value = float("nan")

        valid.append(
            {
                "model": str(row.get("model", "")),
                "ccmr": ccmr_value,
                "ltm": ltm_value,
                "alpha": alpha_value if np.isfinite(alpha_value) else float("nan"),
            }
        )
    return valid


def _ltm_label(valid_rows: list[dict]) -> str:
    alpha_values = sorted(
        {float(r["alpha"]) for r in valid_rows if np.isfinite(float(r["alpha"]))}
    )
    if len(alpha_values) == 1:
        alpha_pct = int(round(alpha_values[0] * 100))
        return f"LTM@{alpha_pct}%"
    return "LTM(CCMR)"


def _padded_positive_limits(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = float(vmax - vmin)
    if span <= 1e-9:
        pad = max(0.1, abs(vmin) * 0.10, 0.05)
    else:
        pad = max(0.1, span * 0.10)
    lo = max(0.0, vmin - pad)
    hi = vmax + pad
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def plot_ccmr_ltm_comparison(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.9, 5.55))
    scatter_ax, bar_ax = axes
    valid_rows = _valid_ccmr_ltm_rows(rows)

    if not valid_rows:
        scatter_ax.set_visible(False)
        bar_ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[scatter_ax, bar_ax])
        return

    label_ltm = _ltm_label(valid_rows)

    # Left: CCMR vs LTM scatter.
    xs = np.asarray([float(r["ccmr"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ltm"]) for r in valid_rows], dtype=float)
    lim_lo, lim_hi = _padded_positive_limits(np.concatenate([xs, ys]))

    _style_axes(scatter_ax)
    scatter_ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        linestyle="--",
        linewidth=1.1,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
    )
    for row in sorted(valid_rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        scatter_ax.scatter(
            [float(row["ccmr"])],
            [float(row["ltm"])],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
    scatter_ax.set_xlim(lim_lo, lim_hi)
    scatter_ax.set_ylim(lim_lo, lim_hi)
    scatter_ax.set_xlabel("CCMR", fontsize=10.5)
    scatter_ax.set_ylabel(label_ltm, fontsize=10.5)
    _set_panel_title(scatter_ax, f"CCMR vs {label_ltm}")

    # Right: sorted CCMR/LTM bars to compare rank and tail-gap by model.
    ranked_rows = sorted(
        valid_rows, key=lambda r: (float(r["ltm"]), str(r["model"])), reverse=True
    )
    model_names = [str(r["model"]) for r in ranked_rows]
    ccmr_vals = np.asarray([float(r["ccmr"]) for r in ranked_rows], dtype=float)
    ltm_vals = np.asarray([float(r["ltm"]) for r in ranked_rows], dtype=float)
    colors = [_color_for_model(model) for model in model_names]
    x = np.arange(len(ranked_rows), dtype=float)
    width = 0.38

    _style_axes(bar_ax, grid_axis="y")
    bar_ax.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.1,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
        alpha=0.75,
    )
    bar_ax.bar(
        x - width / 2.0,
        ccmr_vals,
        width=width,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        label="CCMR",
        zorder=3,
    )
    bar_ax.bar(
        x + width / 2.0,
        ltm_vals,
        width=width,
        color=colors,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.6,
        label=label_ltm,
        zorder=3,
    )
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(model_names, rotation=24, ha="right", rotation_mode="anchor")
    bar_ax.tick_params(axis="x", labelsize=8.8, pad=4)
    bar_ax.set_ylabel("Score", fontsize=10.5)
    _set_panel_title(bar_ax, f"Sorted by {label_ltm}")
    y_lo, y_hi = _padded_positive_limits(np.concatenate([ccmr_vals, ltm_vals]))
    bar_ax.set_ylim(y_lo, y_hi)
    bar_ax.legend(frameon=False, loc="upper right", fontsize=9.0)

    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[scatter_ax, bar_ax],
        wspace=0.20,
        add_legend=False,
        left=0.07,
        right=0.985,
        top=0.90,
        bottom=0.18,
    )


def plot_bio_vs_confounder_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_bio_vs_confounder_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_mari_vs_ri_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_mari_vs_ri_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def _draw_ccmr_vs_mari_scatter(ax, rows: list[dict]) -> None:
    ccmr_rows = [r for r in rows if "ccmr" in r and np.isfinite(float(r["ccmr"]))]
    if not ccmr_rows:
        ax.set_visible(False)
        return

    xs = np.asarray([float(r["mari"]) for r in ccmr_rows], dtype=float)
    ys = np.asarray([float(r["ccmr"]) for r in ccmr_rows], dtype=float)

    _style_axes(ax)
    ax.axhline(y=1.0, linestyle="--", linewidth=1.1, color=REFERENCE_LINE_COLOR, zorder=1)

    for row in sorted(ccmr_rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["mari"])
        y = float(row["ccmr"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )

    ax.set_xlim(*_padded_unit_interval_limits(xs))
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    ax.set_ylim(max(0.0, float(ys.min()) - y_pad), float(ys.max()) + y_pad)
    ax.set_xlabel("MaRI", fontsize=10.5)
    ax.set_ylabel("CCMR", fontsize=10.5)
    _set_panel_title(ax, "CCMR vs MaRI")


def _draw_ccmr_sample_distributions(ax, rows: list[dict]) -> None:
    ccmr_rows = [
        r
        for r in rows
        if "ccmr_samples_path" in r
        and "ccmr_q_alpha" in r
        and np.isfinite(float(r.get("ccmr", float("nan"))))
    ]
    if not ccmr_rows:
        ax.set_visible(False)
        return

    model_data = []
    for row in ccmr_rows:
        path = Path(str(row["ccmr_samples_path"]))
        if not path.exists():
            continue
        values = np.load(path)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        alpha = float(row["ccmr_alpha"])
        alpha_pct = int(round(alpha * 100))
        model_data.append(
            {
                "model": str(row["model"]),
                "values": values,
                "q_alpha": float(row["ccmr_q_alpha"]),
                "alpha": alpha,
                "alpha_pct": alpha_pct,
                "ccmr": float(row["ccmr"]),
                "lt1_frac": float(np.mean(values < 1.0)),
            }
        )

    if not model_data:
        ax.set_visible(False)
        return

    model_data = sorted(
        model_data,
        key=lambda d: (float(d["ccmr"]), str(d["model"])),
        reverse=True,
    )
    all_values = np.concatenate([d["values"] for d in model_data])
    x_min = max(0.0, float(np.nanpercentile(all_values, 1)) - 0.1)
    x_max = float(np.nanpercentile(all_values, 99)) + 0.1
    x_grid = np.linspace(x_min, x_max, 512)

    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    # Shade the fragile region (CCMR < 1.0)
    shade_right = min(1.0, x_max)
    if shade_right > x_min:
        ax.axvspan(x_min, shade_right, color=FRAGILE_SHADE_COLOR, alpha=0.55, zorder=1)
    ax.axvline(
        x=1.0,
        linestyle="--",
        linewidth=1.1,
        color=REFERENCE_LINE_COLOR,
        zorder=2,
        alpha=0.75,
    )

    row_centers = np.arange(len(model_data), 0, -1, dtype=float)
    row_spacing = 1.0
    amplitude = 0.72
    global_peak = 0.0
    rendered_rows: list[tuple[dict, np.ndarray, np.ndarray | None, np.ndarray, float]] = []
    for d in model_data:
        values = d["values"]
        try:
            density = gaussian_kde(values, bw_method="scott")(x_grid)
        except Exception:
            density = None

        if density is None:
            counts, edges = np.histogram(values, bins=40, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            y_values = counts
            x_values = centers
        else:
            y_values = density
            x_values = x_grid
        peak = float(np.nanmax(y_values)) if len(y_values) > 0 else 0.0
        global_peak = max(global_peak, peak)
        rendered_rows.append((d, np.asarray(x_values, dtype=float), density, np.asarray(y_values, dtype=float), peak))

    if global_peak <= 0.0:
        global_peak = 1.0

    for row_center, rendered in zip(row_centers, rendered_rows):
        d, x_values, density, y_values, peak = rendered
        color = _color_for_model(d["model"])
        scale = amplitude / max(peak, 1e-9)
        y_curve = row_center + y_values * scale
        ax.hlines(
            y=row_center,
            xmin=x_min,
            xmax=x_max,
            color=SPINE_COLOR,
            linewidth=0.7,
            alpha=0.7,
            zorder=1,
        )
        ax.fill_between(
            x_values,
            row_center,
            y_curve,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        if density is None:
            ax.step(
                x_values,
                y_curve,
                where="mid",
                color=color,
                linewidth=1.5,
                alpha=0.90,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                y_curve,
                color=color,
                linewidth=1.7,
                alpha=0.95,
                zorder=3,
            )

        q = float(d["q_alpha"])
        if np.isfinite(q) and x_min <= q <= x_max:
            ax.vlines(
                x=q,
                ymin=row_center - 0.18,
                ymax=row_center + amplitude,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.95,
                zorder=4,
            )

        ax.text(
            -0.02,
            row_center,
            str(d["model"]),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10.0,
            color=TEXT_COLOR,
            weight="semibold",
        )
    alpha_pct = int(model_data[0]["alpha_pct"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.5, float(len(model_data)) + 0.95)
    ax.set_xlabel("Per-sample CCMR", fontsize=10.5)
    ax.set_ylabel("")


def plot_ccmr_sample_distributions(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    valid_rows = [
        r
        for r in rows
        if "ccmr_samples_path" in r
        and "ccmr_q_alpha" in r
        and np.isfinite(float(r.get("ccmr", float("nan"))))
    ]
    fig_height = max(5.5, 1.0 + 0.70 * max(1, len(valid_rows)))
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(13.6, fig_height),
        gridspec_kw={"width_ratios": [5.35, 1.45]},
    )
    alpha_values = [
        int(round(float(r["ccmr_alpha"]) * 100))
        for r in valid_rows
        if "ccmr_alpha" in r and np.isfinite(float(r["ccmr_alpha"]))
    ]
    alpha_pct = alpha_values[0] if alpha_values else 10
    _draw_ccmr_sample_distributions(ax, rows)
    info_ax.set_axis_off()
    if ax.get_visible():
        info_ax.set_ylim(ax.get_ylim())
        info_ax.set_xlim(0.0, 1.0)
        info_ax.text(
            0.02,
            float(len(valid_rows)) + 0.72,
            "CCMR",
            ha="left",
            va="center",
            fontsize=9.0,
            color=TEXT_COLOR,
            weight="semibold",
            family="DejaVu Sans Mono",
        )
        info_ax.text(
            0.26,
            float(len(valid_rows)) + 0.72,
            f"Q{alpha_pct}",
            ha="left",
            va="center",
            fontsize=9.0,
            color=TEXT_COLOR,
            weight="semibold",
            family="DejaVu Sans Mono",
        )
        info_ax.text(
            0.47,
            float(len(valid_rows)) + 0.72,
            "%<1",
            ha="left",
            va="center",
            fontsize=9.0,
            color=TEXT_COLOR,
            weight="semibold",
            family="DejaVu Sans Mono",
        )
        ccmr_rows = [
            r
            for r in rows
            if "ccmr_samples_path" in r
            and "ccmr_q_alpha" in r
            and np.isfinite(float(r.get("ccmr", float("nan"))))
        ]
        model_data: list[dict[str, float | str | np.ndarray]] = []
        for row in ccmr_rows:
            path = Path(str(row["ccmr_samples_path"]))
            if not path.exists():
                continue
            values = np.load(path)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            model_data.append(
                {
                    "model": str(row["model"]),
                    "ccmr": float(row["ccmr"]),
                    "q_alpha": float(row["ccmr_q_alpha"]),
                    "lt1_frac": float(np.mean(values < 1.0)),
                }
            )
        model_data = sorted(
            model_data,
            key=lambda d: (float(d["ccmr"]), str(d["model"])),
            reverse=True,
        )
        row_centers = np.arange(len(model_data), 0, -1, dtype=float)
        for row_center, row in zip(row_centers, model_data):
            info_ax.text(
                0.02,
                float(row_center),
                f"{float(row['ccmr']):.3f}",
                ha="left",
                va="center",
                fontsize=9.2,
                color=TEXT_COLOR,
                family="DejaVu Sans Mono",
            )
            info_ax.text(
                0.26,
                float(row_center),
                f"{float(row['q_alpha']):.3f}",
                ha="left",
                va="center",
                fontsize=9.2,
                color=TEXT_COLOR,
                family="DejaVu Sans Mono",
            )
            info_ax.text(
                0.47,
                float(row_center),
                f"{100.0 * float(row['lt1_frac']):.1f}%",
                ha="left",
                va="center",
                fontsize=9.2,
                color=TEXT_COLOR,
                family="DejaVu Sans Mono",
            )
    fig.suptitle(
        "Per-sample CCMR distributions",
        fontsize=15,
        weight="semibold",
        y=0.982,
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        0.928,
        f"(sorted by CCMR; dotted: $Q_{{{alpha_pct}}}$; shaded: CCMR < 1)",
        ha="center",
        va="center",
        fontsize=10.0,
        color=REFERENCE_LINE_COLOR,
    )
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax, info_ax],
        add_legend=False,
        left=0.165,
        right=0.94,
        top=0.885,
        bottom=0.095,
        wspace=0.01,
    )


def plot_ccmr_vs_mari_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_ccmr_vs_mari_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def _draw_q_alpha_vs_ccmr_scatter(ax, rows: list[dict]) -> None:
    valid_rows = [
        r
        for r in rows
        if "ccmr" in r
        and "ccmr_q_alpha" in r
        and np.isfinite(float(r["ccmr"]))
        and np.isfinite(float(r["ccmr_q_alpha"]))
    ]
    if not valid_rows:
        ax.set_visible(False)
        return

    alpha_pct_values = [
        int(round(float(r["ccmr_alpha"]) * 100))
        for r in valid_rows
        if "ccmr_alpha" in r and np.isfinite(float(r["ccmr_alpha"]))
    ]
    alpha_pct = alpha_pct_values[0] if alpha_pct_values else 10

    xs = np.asarray([float(r["ccmr"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ccmr_q_alpha"]) for r in valid_rows], dtype=float)

    _style_axes(ax)
    ax.axhline(y=1.0, linestyle="--", linewidth=1.1, color=REFERENCE_LINE_COLOR, zorder=1)
    ax.axvline(x=1.0, linestyle="--", linewidth=1.1, color=REFERENCE_LINE_COLOR, zorder=1)

    for row in sorted(valid_rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["ccmr"])
        y = float(row["ccmr_q_alpha"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )

    all_vals = np.concatenate([xs, ys])
    x_pad = max(0.1, (xs.max() - xs.min()) * 0.10) if xs.size > 0 else 0.5
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    ax.set_xlim(max(0.0, float(xs.min()) - x_pad), float(xs.max()) + x_pad)
    ax.set_ylim(max(0.0, float(ys.min()) - y_pad), float(ys.max()) + y_pad)
    ax.set_xlabel("CCMR", fontsize=10.5)
    ax.set_ylabel(f"Q{alpha_pct}", fontsize=10.5)
    _set_panel_title(ax, f"Q{alpha_pct} vs CCMR")


def plot_q_alpha_vs_ccmr_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_q_alpha_vs_ccmr_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)