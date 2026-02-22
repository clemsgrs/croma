from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_rank(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda r: float(r["mari"]), reverse=True)
    models = [r["model"] for r in sorted_rows]
    ri_vals = [float(r["ri"]) for r in sorted_rows]
    mari_vals = [float(r["mari"]) for r in sorted_rows]

    y = np.arange(len(models), dtype=float)
    h = 0.35
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(models) + 1.8)))
    ax.set_facecolor("#fbfcfd")
    ax.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.barh(y - h / 2.0, ri_vals, height=h, color="#5086a9", label="RI")
    ax.barh(y + h / 2.0, mari_vals, height=h, color="#db6f4d", label="MaRI")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Robustness score", fontsize=11)
    ax.set_title("RI / MaRI Ranking", fontsize=14, weight="bold")
    ax.legend(frameon=False, loc="lower right")

    for yi, rv, mv in zip(y, ri_vals, mari_vals):
        ax.text(rv + 0.01, yi - h / 2.0, f"{rv:.3f}", va="center", fontsize=9, color="#2f3b45")
        ax.text(mv + 0.01, yi + h / 2.0, f"{mv:.3f}", va="center", fontsize=9, color="#2f3b45")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ri_vs_mari(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    xs = np.asarray([float(r["ri"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari"]) for r in rows], dtype=float)
    names = [str(r["model"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.scatter(xs, ys, s=90, color="#2a9d8f", edgecolors="white", linewidths=1.0, zorder=3)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.2, color="#6b7280", zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("RI", fontsize=11)
    ax.set_ylabel("MaRI", fontsize=11)
    ax.set_title("RI vs MaRI", fontsize=14, weight="bold")

    offsets = [(-10, 9), (8, 10), (-8, -12), (10, -10), (-12, 2), (12, 1)]
    for i, name in enumerate(names):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            name,
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            fontsize=9,
            color="#2f3b45",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tail_fragility(rows: list[dict], out_path: Path, alpha: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xs = np.asarray([float(r["mari_q_alpha"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari_ltm_alpha"]) for r in rows], dtype=float)
    names = [str(r["model"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.add_patch(Rectangle((0.0, 0.0), 0.35, 0.35, facecolor="#f7d7c4", alpha=0.28, zorder=0))
    ax.scatter(xs, ys, s=90, color="#e76f51", edgecolors="white", linewidths=1.0, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"MaRI Q{alpha:g}", fontsize=11)
    ax.set_ylabel(f"MaRI LTM{alpha:g}", fontsize=11)
    ax.set_title("Tail Fragility (MaRI)", fontsize=14, weight="bold")

    offsets = [(-10, 9), (8, 10), (-8, -12), (10, -10), (-12, 2), (12, 1)]
    for i, name in enumerate(names):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            name,
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            fontsize=9,
            color="#2f3b45",
        )

    ax.text(
        0.02,
        0.02,
        "Fragile zone",
        transform=ax.transAxes,
        fontsize=9,
        color="#7f1d1d",
        ha="left",
        va="bottom",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

