"""Plot 1 -- CRoMa vs pretraining scale (#WSIs), PathoROB-style scatter (#63).

A single scatter over the 16-model tile panel that asks whether *pretraining
scale* predicts robustness. The empirical direction is NOT assumed here: a weak
or absent relationship is itself the message, so the figure only *shows* the
points (plus a direction-neutral Spearman summary) and lets the reader judge.

Encoding
--------
- ``x``          : disclosed #WSIs on a **log** axis (metadata ``n_wsis``; four
                   values are PathoROB-filled and flagged in ``wsis_source``).
- ``y``          : each model's pooled headline CRoMa at ``m=5``, averaged across
                   the three tile benchmarks {Camelyon, TCGA-4x4, Tolkach} -- the
                   same cross-dataset quantity the cross-benchmark figure uses
                   (the ``croma`` column of each benchmark's
                   ``results/metrics.csv``).
- ``colour``     : regime, ``VLFM`` vs ``vision-only`` (croma.plotstyle.REGIME_COLOR).
- ``marker area``: proportional to log-params (metadata ``params``); models whose
                   parameter count is undisclosed are drawn hollow.

Separation of concerns
----------------------
:func:`build_scale_frame` is a PURE function of the metadata frame and a
per-dataset CRoMa mapping -- it never touches the filesystem, so it is trivially
testable and runs anywhere. :func:`main` loads the per-dataset CRoMa from the
(gitignored) benchmark outputs at runtime *behind a data-availability guard*,
then calls :func:`build_scale_frame` and renders. When the output tree is absent
(e.g. a fresh checkout) the module still imports and ``build_scale_frame`` still
runs on provided data; only the real render is skipped.

Run: python scripts/repro/figures/scale_scatter.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe; figure scripts never show()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Repo root: scripts/repro/figures/scale_scatter.py -> parents[3].
REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Identity only from croma.plotstyle (registers fonts + rcParams on import). We do
# NOT import scripts/bench/plotting.py so this figure stays self-contained.
from croma import plotstyle  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_METADATA = HERE.parent / "model_metadata.csv"
DEFAULT_OUT = REPO / "paper" / "figures" / "scale_scatter.pdf"

# The three tile benchmarks whose ``croma`` column is averaged into the y-axis.
# Each entry lists candidate run dirs (first existing wins), mirroring the loading in
# ``cross_benchmark_figure.py`` on the output/metrics/<protocol>/<benchmark> layout.
BENCHMARKS: list[tuple[str, tuple[str, ...]]] = [
    ("Camelyon", ("output/metrics/k-star/pathorob-camelyon",)),
    ("TCGA-4x4", ("output/metrics/k-star/pathorob-tcga-4x4",)),
    ("Tolkach", ("output/metrics/k-star/pathorob-tolkach-esca",)),
]

#: Models excluded from this figure by construction, not by missing data. The x-axis is
#: pretraining #WSIs; a natural-image control never saw a slide, so it has no position on
#: that axis -- ``n/a``, not ``n/d``. Dropping it here (loudly) rather than letting its
#: NaN x silently vanish into the log axis keeps the exclusion a stated choice that the
#: caption can name.
CONTROL_MODELS = ("DINOv2-B",)

# LaTeX-scale unit suffixes -> multiplier.
_UNIT_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}
_MISSING_TOKENS = {"", "nan", "n/d", "n/a", "--", "none"}


# ---------------------------------------------------------------------------
# Pure parsing / assembly (no filesystem).
# ---------------------------------------------------------------------------
def _parse_scale(cell: object) -> float:
    """Parse a LaTeX-decorated count into a raw float.

    Handles the metadata's ``params``/``n_wsis`` cells, e.g. ``$3.1$M`` ->
    ``3.1e6``, ``${\\sim}1.1$B`` -> ``1.1e9``, ``${>}1$M`` -> ``1e6``,
    ``$6{,}093$`` -> ``6093``. Undisclosed cells (``n/d`` and friends) -> ``nan``.
    """
    text = str(cell).strip()
    if text.lower() in _MISSING_TOKENS:
        return float("nan")
    cleaned = re.sub(r"\\[a-zA-Z]+", "", text)  # drop LaTeX commands (\sim, \code...)
    for ch in "${}~<>\\ ":
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.replace(",", "")  # thousands separator, e.g. 6,093
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([kKmMbB]?)", cleaned)
    if not match:
        return float("nan")
    return float(match.group(1)) * _UNIT_MULT.get(match.group(2).lower(), 1.0)


def normalize_regime(value: object) -> str:
    """Collapse a regime cell to the two-way label used for colour.

    ``vision--language`` / ``VLFM`` -> ``"VLFM"``; everything else -> ``"vision-only"``.
    """
    key = str(value).strip().lower()
    if key == "vlfm" or "language" in key:
        return "VLFM"
    return "vision-only"


def build_scale_frame(
    metadata: pd.DataFrame,
    per_dataset_croma: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Assemble the per-model scatter frame (PURE; no filesystem).

    Parameters
    ----------
    metadata:
        The model-metadata frame (as loaded from ``model_metadata.csv``). Only the
        ``panel == "tile"`` rows are used -- slide-level models are excluded, as are
        the natural-image controls in ``CONTROL_MODELS`` (no #WSIs axis position).
    per_dataset_croma:
        ``{dataset_name: {model: croma}}``. Each model's y-value is the mean of its
        CRoMa across *all* provided datasets; a model missing (or NaN) in any one of
        them is dropped, mirroring the cross-benchmark figure's "present in all".

    Returns
    -------
    One row per included tile model with columns ``model``, ``croma_mean`` (the
    cross-dataset mean at ``m=5``), ``regime`` (normalised), ``params`` and
    ``n_wsis`` (parsed to raw floats; ``params`` may be NaN when undisclosed) and
    ``n_datasets`` (the number averaged).
    """
    datasets = list(per_dataset_croma.keys())
    tile = metadata[metadata["panel"] == "tile"]
    tile = tile[~tile["model"].isin(CONTROL_MODELS)]
    rows: list[dict[str, object]] = []
    for _, meta in tile.iterrows():
        model = str(meta["model"])
        values: list[float] = []
        complete = True
        for name in datasets:
            croma = per_dataset_croma[name].get(model)
            if croma is None or pd.isna(croma):
                complete = False
                break
            values.append(float(croma))
        if not complete or not values:
            continue
        rows.append(
            {
                "model": model,
                "croma_mean": sum(values) / len(values),
                "regime": normalize_regime(meta["regime"]),
                "params": _parse_scale(meta["params"]),
                "n_wsis": _parse_scale(meta["n_wsis"]),
                "n_datasets": len(values),
            }
        )
    return pd.DataFrame(
        rows, columns=["model", "croma_mean", "regime", "params", "n_wsis", "n_datasets"]
    )


# ---------------------------------------------------------------------------
# Data loading (filesystem; guarded).
# ---------------------------------------------------------------------------
def load_metadata(path: Path = DEFAULT_METADATA) -> pd.DataFrame:
    """Load the committed model metadata (empty cells normalised to ``""``).

    ``keep_default_na=False`` keeps the literal ``n/a`` cells (not applicable) distinct
    from truly empty ones; ``_parse_scale`` maps both to NaN anyway, but the distinction
    matters to anything that reads the raw cell.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def _metrics_path(repo: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        candidate = Path(repo) / rel / "results" / "metrics.csv"
        if candidate.exists():
            return candidate
    return None


def load_per_dataset_croma(repo: Path = REPO) -> dict[str, dict[str, float]] | None:
    """Load ``{dataset: {model: croma}}`` from the three benchmark outputs.

    Returns ``None`` when any benchmark's ``metrics.csv`` is absent (the caller's
    data-availability guard), so a fresh checkout skips the render cleanly.
    """
    per: dict[str, dict[str, float]] = {}
    for name, candidates in BENCHMARKS:
        path = _metrics_path(repo, candidates)
        if path is None:
            return None
        series = pd.read_csv(path).set_index("model")["croma"]
        per[name] = {str(k): float(v) for k, v in series.items()}
    return per


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def _marker_sizes(
    params: np.ndarray, *, smin: float = 45.0, smax: float = 340.0
) -> tuple[np.ndarray, np.ndarray]:
    """Map params -> marker *area* on a log scale; NaN params -> smallest size.

    Returns ``(sizes, known)`` where ``known`` flags the models whose parameter
    count is disclosed (drawn filled; the rest are drawn hollow).
    """
    values = np.asarray(params, dtype=float)
    known = np.isfinite(values) & (values > 0)
    sizes = np.full(values.shape, smin, dtype=float)
    if known.any():
        logp = np.log10(values[known])
        spread = logp.max() - logp.min()
        norm = (logp - logp.min()) / spread if spread > 0 else np.zeros_like(logp)
        sizes[known] = smin + norm * (smax - smin)
    return sizes, known


def render_scale_scatter(frame: pd.DataFrame, out_path: Path = DEFAULT_OUT) -> Path:
    """Draw the scale-vs-robustness scatter and write PDF (+ PNG sibling)."""
    plotstyle.apply_style()
    sizes, known = _marker_sizes(frame["params"].to_numpy())

    fig, ax = plt.subplots(figsize=(plotstyle.COL_ONEHALF, plotstyle.COL_ONEHALF * 0.86))
    plotstyle.style_axes(ax)
    ax.set_xscale("log")
    ax.axhline(
        0.0,
        color=plotstyle.REFERENCE_LINE_COLOR,
        lw=plotstyle.LW_REFERENCE,
        ls="--",
        zorder=1,
    )

    models = frame["model"].tolist()
    xs = frame["n_wsis"].to_numpy(dtype=float)
    ys = frame["croma_mean"].to_numpy(dtype=float)
    for i, model in enumerate(models):
        regime = frame["regime"].iloc[i]
        color = plotstyle.REGIME_COLOR.get(regime, plotstyle.REGIME_COLOR["vision-only"])
        ax.scatter(
            xs[i],
            ys[i],
            s=sizes[i],
            facecolor=color if known[i] else "white",
            edgecolor=color,
            linewidths=1.1,
            alpha=0.9,
            zorder=3,
        )
        ax.annotate(
            model,
            (xs[i], ys[i]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=plotstyle.FS_ANNOT,
            color=plotstyle.TEXT_COLOR,
        )

    ax.set_xlabel("Disclosed pretraining scale (#WSIs, log)")
    ax.set_ylabel(rf"Cross-dataset CRoMa ($m{{=}}{int(CROMA_HEADLINE_M)}$)")
    plotstyle.title_with_subtitle(
        ax,
        "Robustness vs pretraining scale",
        r"marker area $\propto\log$(params); colour = regime",
    )

    # Direction-neutral summary: report the rank correlation, do not assume its sign.
    _annotate_spearman(ax, xs, ys)
    _add_legend(ax)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.13)
    out_path = Path(out_path)
    (out_path.parent / "png").mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path.parent / "png" / out_path.with_suffix(".png").name,
        dpi=plotstyle.DEFAULT_DPI,
    )
    fig.savefig(out_path)  # flat pdf for \graphicspath
    plt.close(fig)
    return out_path


def _annotate_spearman(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    """Annotate Spearman rho of log(#WSIs) vs CRoMa (skipped if scipy absent)."""
    if len(xs) < 3:
        return
    try:
        from scipy.stats import spearmanr
    except Exception:  # noqa: BLE001 - annotation is best-effort
        return
    rho, _ = spearmanr(np.log10(xs), ys)
    ax.text(
        0.03,
        0.05,
        rf"Spearman $\rho = {rho:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=plotstyle.FS_ANNOT,
        color=plotstyle.MUTED_TEXT_COLOR,
    )


def _add_legend(ax) -> None:
    handles = [
        plt.Line2D(
            [0], [0], marker="o", ls="", ms=7,
            mfc=plotstyle.REGIME_COLOR["VLFM"], mec=plotstyle.REGIME_COLOR["VLFM"],
            label="VLFM",
        ),
        plt.Line2D(
            [0], [0], marker="o", ls="", ms=7,
            mfc=plotstyle.REGIME_COLOR["vision-only"],
            mec=plotstyle.REGIME_COLOR["vision-only"], label="vision-only",
        ),
        plt.Line2D(
            [0], [0], marker="o", ls="", ms=7, mfc="white",
            mec=plotstyle.MUTED_TEXT_COLOR, mew=1.1, label="params undisclosed",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=plotstyle.FS_ANNOT,
        handletextpad=0.4,
        labelspacing=0.3,
    )


# ---------------------------------------------------------------------------
# Generator entry point.
# ---------------------------------------------------------------------------
def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)
    metadata = load_metadata(args.metadata)
    per_dataset = load_per_dataset_croma(args.repo_root)
    if per_dataset is None:
        print(
            "scale scatter: benchmark CRoMa outputs not found under "
            f"{args.repo_root}/output/ -- the frame builder is available but the "
            "render was skipped. Run where the (gitignored) output/ tree exists."
        )
        return 0

    frame = build_scale_frame(metadata, per_dataset)
    print(f"scale scatter: {len(frame)} tile models over {list(per_dataset)}")
    print(frame.to_string(index=False))
    out = render_scale_scatter(frame, args.out)
    print(f"\nwrote {out} and {out.parent / 'png' / out.with_suffix('.png').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
