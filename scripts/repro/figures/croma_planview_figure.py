"""Plan-view of the PCaBiop typed distances in the ``(d_SO, d_OS)`` plane (supp:geometry).

Companion panel to the analytic schematic in the supplement's geometry section.
The schematic states that ``CRoMa`` reads only the *direction* of the typed-distance
pair; this panel shows what that plane actually looks like for the four slide-level
encoders, and why the scale-free construction is not a cosmetic choice: their median
typed-distance magnitudes span roughly two orders of magnitude, so the radial
coordinate is pure encoder nuisance.

Encoding
--------
- ``x`` / ``y`` : per-slide ``d_SO`` / ``d_OS`` (mean cosine distance to the ``m``
                 nearest same-biology/other-centre and other-biology/same-centre
                 neighbours) on **log** axes -- the panel's 80x radial spread makes a
                 linear plane unreadable. Constant-``CRoMa`` loci stay lines parallel
                 to the diagonal under the log map, so "offset from the diagonal =
                 margin" survives; "ray from the origin" becomes "translation along
                 the diagonal".
- ``colour``   : model (croma.plotstyle.color_for_model), direct-labelled, no legend.
- ``contour``  : a per-model density outline, drawn because the two clouds that
                 overlap here -- PRISM and TITAN, the panel's whole point -- carry
                 adjacent family hues whose normal-vision separation sits just under
                 the categorical floor. The outline is the secondary encoding that
                 keeps them separable without departing from the house palette.
- ``dashed``   : the ``d_SO = d_OS`` diagonal, i.e. ``CRoMa = 0``.

Why the distances are recomputed here
-------------------------------------
``CRoMa`` persists only the ratio-derived margin: ``_compute_sample_croma`` reduces
``(d_SO, d_OS)`` to ``(d_OS - d_SO) / (d_OS + d_SO)`` and the pair is discarded, so no
artefact under ``<run>/results`` carries the plane coordinates. This script therefore
re-runs the *same* typed-neighbour search the metric uses --
``croma.metrics.croma._iterative_typed_neighbor_search``, imported rather than
reimplemented -- and keeps the means. :func:`typed_distance_frame` then asserts the
implied margins reproduce the published per-sample ``croma_m5`` column, so the panel
can never silently drift from the tables it illustrates.

Run: python scripts/repro/figures/croma_planview_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe; figure scripts never show()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Repo root: scripts/repro/figures/croma_planview_figure.py -> parents[3].
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "src", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting import style as plotstyle  # noqa: E402
from croma.metrics.croma import _iterative_typed_neighbor_search  # noqa: E402
from paper_manifest import by_prefix  # noqa: E402

PANDA = by_prefix("Panda")
EMBEDDINGS_REL = "output/embeddings/panda-wsi"
BENCHMARK_REL = "data/benchmarks/panda.csv"

#: Neighbourhood radius and search schedule -- the headline operating point, matching
#: the params recorded in the run's cache index.
M = 5
START_K = 200
K_GROWTH = 2.0

#: Bottom-left to top-right, i.e. increasing typed-distance magnitude. Fixes the
#: draw order so overplotting is deterministic across renders.
MODELS = ("Prov-GigaPath", "TITAN", "PRISM", "MOOZY")

#: Direct-label offsets (points), hand-placed: the clouds are dense and unevenly
#: shaped, so automatic placement collides with either the diagonal or a neighbour.
LABEL_OFFSETS = {
    "Prov-GigaPath": (11, -12),
    "TITAN": (12, -10),
    "PRISM": (-12, 12),
    "MOOZY": (-11, 12),
}
LABEL_ALIGN = {
    "Prov-GigaPath": "left",
    "TITAN": "left",
    "PRISM": "right",
    "MOOZY": "right",
}

#: Encoders whose median gets a perpendicular drop onto the diagonal. Restricted to
#: the pair sitting at nearly the same radius on opposite sides of it -- the panel's
#: central claim, and the only one position alone does not already make obvious.
MARGIN_DROP_MODELS = ("PRISM", "TITAN")


def typed_distance_frame(
    *,
    embeddings_dir: Path,
    benchmark_csv: Path,
    per_sample_dir: Path,
    models: tuple[str, ...] = MODELS,
    m: int = M,
) -> pd.DataFrame:
    """Per-slide ``(d_SO, d_OS)`` for every model, validated against the run.

    Returns one row per (model, slide) with columns ``model``, ``d_so``, ``d_os``
    and ``croma``. Raises ``AssertionError`` if the implied margin disagrees with
    the published ``croma_m5`` for any sample -- that mismatch would mean this panel
    and Table~\\ref{tab:main-results-panda} describe different geometry.
    """
    manifest = pd.read_csv(embeddings_dir / "manifest.csv")
    bench = pd.read_csv(benchmark_csv)
    row_of = {s: i for i, s in enumerate(manifest["sample_id"])}
    missing = set(bench["sample_id"]) - set(row_of)
    if missing:
        raise KeyError(f"{len(missing)} benchmark ids absent from the embedding manifest")

    rows = bench["sample_id"].map(row_of).to_numpy(dtype=int)
    labels = pd.factorize(bench["label"])[0].astype(int)
    centers = pd.factorize(bench["data_provider"])[0].astype(int)
    slide_ids = bench["slide_id"].astype(str).to_numpy()

    frames = []
    for model in models:
        features = np.load(embeddings_dir / f"{model}.npy")[rows].astype(np.float64)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12

        so_dists, os_dists, _ = _iterative_typed_neighbor_search(
            features=features,
            labels=labels,
            centers=centers,
            slide_ids=slide_ids,
            m=int(m),
            start_k=START_K,
            k_growth_factor=K_GROWTH,
        )
        undefined = np.any(np.isinf(so_dists), axis=1) | np.any(np.isinf(os_dists), axis=1)
        d_so = np.where(undefined, np.nan, so_dists.mean(axis=1))
        d_os = np.where(undefined, np.nan, os_dists.mean(axis=1))
        croma = (d_os - d_so) / (d_os + d_so)

        published = (
            pd.read_csv(per_sample_dir / f"{model}.csv")
            .sort_values("sample_index")[f"croma_m{m}"]
            .to_numpy()
        )
        delta = np.nanmax(np.abs(croma - published))
        assert delta < 1e-3, f"{model}: recomputed CRoMa drifts from the run by {delta:.2e}"

        frames.append(pd.DataFrame({"model": model, "d_so": d_so, "d_os": d_os, "croma": croma}))

    return pd.concat(frames, ignore_index=True).dropna(subset=["d_so", "d_os"])


#: Density-outline level: the contour enclosing this share of a model's slides.
CONTOUR_MASS = 0.75


def _density_outline(ax, x: np.ndarray, y: np.ndarray, *, color: str) -> None:
    """Trace the contour enclosing ``CONTOUR_MASS`` of a model's slides.

    Estimated in log space, because that is the space the axes display: a Gaussian
    kernel on the raw distances would be lopsided once the axis takes the log.
    """
    from scipy.stats import gaussian_kde

    lx, ly = np.log10(x), np.log10(y)
    kde = gaussian_kde(np.vstack([lx, ly]))

    pad = 0.45
    gx, gy = np.mgrid[
        lx.min() - pad : lx.max() + pad : 120j,
        ly.min() - pad : ly.max() + pad : 120j,
    ]
    density = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    # The level enclosing CONTOUR_MASS of the probability mass: walk the sorted
    # densities until the cumulative share crosses the target.
    flat = np.sort(density.ravel())[::-1]
    share = np.cumsum(flat) / flat.sum()
    level = float(flat[np.searchsorted(share, CONTOUR_MASS)])

    ax.contour(
        10**gx, 10**gy, density, levels=[level],
        colors=[color], linewidths=0.9, alpha=0.85, zorder=4,
    )


def _margin_drop(ax, x: float, y: float, *, color: str) -> None:
    """Drop a hairline from ``(x, y)`` onto the diagonal, perpendicular in log space.

    The foot of the perpendicular from ``(log x, log y)`` onto the line ``v = u`` is
    their midpoint, i.e. the geometric mean ``sqrt(x y)`` back in data coordinates,
    and the segment has length ``|log(y / x)| / sqrt(2)``. That is the log-space image
    of the angular offset ``theta - pi/4`` the schematic draws in the linear plane:
    the log map sends constant-``CRoMa`` rays to lines parallel to the diagonal, so
    the angle becomes a displacement. It reads as a true perpendicular on screen only
    because the axes are equal-aspect over identical limits, which puts the diagonal
    at 45 degrees.
    """
    foot = float(np.sqrt(x * y))
    ax.plot(
        [x, foot], [y, foot],
        color=color, lw=0.7, alpha=0.9, solid_capstyle="butt", zorder=4,
    )


def render(frame: pd.DataFrame, out_path: Path) -> Path:
    """Draw the plan-view; PNG at ``out_path``, PDF in the sibling ``pdf/`` dir.

    Mirrors the studies-plot layout the other curated figures use, so
    ``check_paper_figures`` finds the PDF where it expects it. Returns the PDF path.
    """
    plotstyle.apply_style()
    # Drawn at its final printed size: the supplement includes this at
    # width=0.54\linewidth (~3.5in on a 6.5in text block), so a 1:1 include keeps
    # the labels at the plotstyle point sizes instead of rescaling them.
    fig, ax = plt.subplots(figsize=(3.50, 3.50))

    finite = frame[["d_so", "d_os"]].to_numpy()
    lo = float(np.nanmin(finite)) / 1.5
    hi = float(np.nanmax(finite)) * 1.35

    ax.plot(
        [lo, hi], [lo, hi],
        ls="--", lw=plotstyle.LW_REFERENCE, color=plotstyle.REFERENCE_LINE_COLOR,
        zorder=2, solid_capstyle="butt",
    )

    for model in MODELS:
        g = frame[frame.model == model]
        color = plotstyle.color_for_model(model)
        ax.scatter(g.d_so, g.d_os, s=1.8, c=color, alpha=0.18, lw=0, zorder=3, rasterized=True)
        _density_outline(ax, g.d_so.to_numpy(), g.d_os.to_numpy(), color=color)
        mx, my = float(g.d_so.median()), float(g.d_os.median())
        if model in MARGIN_DROP_MODELS:
            _margin_drop(ax, mx, my, color=color)
        ax.scatter([mx], [my], s=24, c=color, edgecolor="white", lw=0.7, zorder=5)
        ax.annotate(
            model, (mx, my),
            textcoords="offset points", xytext=LABEL_OFFSETS[model],
            ha=LABEL_ALIGN[model], va="center",
            fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR, zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$d^{\mathrm{SO}}_{m}$", fontsize=plotstyle.FS_LABEL)
    ax.set_ylabel(r"$d^{\mathrm{OS}}_{m}$", fontsize=plotstyle.FS_LABEL)

    # Side of the diagonal, stated once, in the corners the clouds never reach.
    ax.text(0.035, 0.955, "robust", transform=ax.transAxes, ha="left", va="top",
            fontsize=plotstyle.FS_ANNOT, style="italic", color=plotstyle.MUTED_TEXT_COLOR)
    ax.text(0.965, 0.045, "fragile", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=plotstyle.FS_ANNOT, style="italic", color=plotstyle.MUTED_TEXT_COLOR)

    plotstyle.style_axes(ax)
    fig.tight_layout(pad=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.parent / "pdf" / f"{out_path.stem}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plotstyle.DEFAULT_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    embeddings_dir = REPO / EMBEDDINGS_REL
    per_sample_dir = REPO / PANDA.run_rel / "results" / "per_sample_metrics_by_model"
    if not embeddings_dir.exists() or not per_sample_dir.exists():
        print(
            f"skipped: {embeddings_dir} or {per_sample_dir} absent -- "
            "run scripts/repro/run_benchmarks.sh first.",
            file=sys.stderr,
        )
        return

    frame = typed_distance_frame(
        embeddings_dir=embeddings_dir,
        benchmark_csv=REPO / BENCHMARK_REL,
        per_sample_dir=per_sample_dir,
    )
    radii = frame.assign(r=np.hypot(frame.d_so, frame.d_os)).groupby("model")["r"].median()
    print(f"median radius spread: {radii.max() / radii.min():.0f}x")

    out_png = REPO / PANDA.studies_rel / "plots" / "croma_planview.png"
    print(f"wrote {render(frame, out_png)}")


if __name__ == "__main__":
    main()
