"""Plot 1 -- CRoMa vs pretraining scale (#WSIs), PathoROB-style scatter (#63).

A single scatter over the pathology tile-model panel (the natural-image control is
excluded; see ``CONTROL_MODELS``) that asks whether *pretraining scale* predicts
robustness. The empirical direction is NOT assumed here: a weak
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
for _p in (REPO / "src", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Identity only from croma.plotstyle (registers fonts + rcParams on import). We do
# NOT import scripts/bench/plotting.py so this figure stays self-contained.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting import style as plotstyle  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402

# The run directory backing each benchmark -- and thus the protocol -- is owned by
# paper_manifest, never spelled out here. This figure once hard-coded k-star run dirs; when
# the tile panel was re-run at median-k those runs were archived, so it silently read a
# directory that no longer existed and skipped its render. Ask the manifest instead.
from paper_manifest import by_benchmark  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_METADATA = HERE.parents[1] / "bench" / "model_metadata.csv"
# Rendered beside the study data it reads, under plots/{pdf,png}/ -- the same convention as
# cross_benchmark_figure.py. Nothing writes into paper/figures/; a human copies the PDF a
# float earns, and check_paper_figures.py reports when that copy falls behind this render.
DEFAULT_OUT = REPO / "output" / "studies" / "scale-scatter" / "plots" / "scale_scatter.pdf"

# The three tile benchmarks whose ``croma`` column is averaged into the y-axis, as
# (display name, manifest benchmark key). The run directory -- and thus the protocol
# (median-k for the tile panel) -- comes from paper_manifest, so it is never named here.
BENCHMARKS: list[tuple[str, str]] = [
    ("Camelyon", "pathorob-camelyon"),
    ("TCGA-4x4", "pathorob-tcga-4x4"),
    ("Tolkach", "pathorob-tolkach-esca"),
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
    # Registry identities on disk, published names on the figure.
    return plotstyle.published_models(pd.read_csv(path, keep_default_na=False, na_values=[]))


def load_per_dataset_croma(repo: Path = REPO) -> dict[str, dict[str, float]] | None:
    """Load ``{dataset: {model: croma}}`` from the three tile benchmark runs.

    Each benchmark's run directory is resolved through ``paper_manifest`` at the protocol the
    paper reports (median-k for the tile panel), so no run directory is named here. Returns
    ``None`` when any benchmark's ``metrics.csv`` is absent (the caller's data-availability
    guard), so a fresh checkout skips the render cleanly.
    """
    per: dict[str, dict[str, float]] = {}
    for name, benchmark in BENCHMARKS:
        path = Path(repo) / by_benchmark(benchmark).metrics_rel
        if not path.exists():
            return None
        series = plotstyle.published_models(pd.read_csv(path)).set_index("model")["croma"]
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


#: Cosmetic label placement only. By default a model's label sits just above its marker; an
#: entry here sends it in another direction to break an overprint in the crowded mid-panel.
#: This is a rendering hint keyed by name for legibility -- not a roster: a model not listed
#: falls back to "above", and one absent from the run is simply never drawn.
_LABEL_DIR: dict[str, str] = {
    "CONCHv1.5": "left",
    "GPFM": "left",
    "Prov-GigaPath": "below",
    "H-optimus-0": "below",
    "Hibou-B": "right",
    "Prost40M": "right",
}


def _label_offset(direction: str, radius_pts: float, pad: float = 4.0):
    """Offset ``(dx, dy)`` in points and ``(ha, va)`` placing a label just outside a marker.

    The gap is measured from the marker *edge* (its radius in points), so labels hug markers
    of every size by the same visual margin instead of a fixed centre offset that a large
    marker would swallow.
    """
    gap = radius_pts + pad
    return {
        "above": (0.0, gap, "center", "bottom"),
        "below": (0.0, -gap, "center", "top"),
        "left": (-gap, 0.0, "right", "center"),
        "right": (gap, 0.0, "left", "center"),
    }[direction]


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
        radius_pts = float(np.sqrt(sizes[i] / np.pi))
        dx, dy, ha, va = _label_offset(_LABEL_DIR.get(model, "above"), radius_pts)
        ax.annotate(
            model,
            (xs[i], ys[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=plotstyle.FS_ANNOT,
            color=plotstyle.TEXT_COLOR,
        )

    # Headroom so the top marker's label clears the subtitle and the fanned-out and
    # below-set labels stay inside the axes; the log x-axis gets a little side padding so
    # the leftmost and rightmost labels do not run into the spines.
    ax.set_xlim(10**3.05, 10**6.85)
    ax.set_ylim(float(ys.min()) - 0.05, float(ys.max()) + 0.055)

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
    pdf_path = out_path.parent / "pdf" / out_path.name
    png_path = out_path.parent / "png" / out_path.with_suffix(".png").name
    for sub in (pdf_path, png_path):
        sub.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=plotstyle.DEFAULT_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def _annotate_spearman(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    """Annotate Spearman rho of log(#WSIs) vs CRoMa (skipped if scipy absent)."""
    if len(xs) < 3:
        return
    try:
        from scipy.stats import spearmanr
    except Exception:  # noqa: BLE001 - annotation is best-effort
        return
    rho, _ = spearmanr(np.log10(xs), ys)
    # Lower-right corner: the bottom-left is occupied by Prost40M and the top by Midnight-12k,
    # so the empty high-#WSIs / low-CRoMa corner is the one that collides with nothing.
    ax.text(
        0.98,
        0.04,
        rf"Spearman $\rho = {rho:.2f}$",
        transform=ax.transAxes,
        ha="right",
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
    png = out.parent.parent / "png" / out.with_suffix(".png").name
    print(f"\nwrote {out} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
