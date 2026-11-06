"""Every label on a Pareto panel sits clear of every point and every other label.

Both Pareto panels name each encoder in place: bold tags for the frontier, small muted labels
for the rest. On a 27-encoder roster the robust corner is crowded -- three frontier rings within
a few points of each other on Camelyon, two on the rank overview -- and a tag laid over a
neighbour's ring or dot hides the very point it should help identify.

These tests read the geometry back from the rendered figure, not from the layout's own box
estimates: each label's drawn extent against each marker and ring disk, and against every other
label. They read it in the renderers that write the exported files -- the 300-dpi PNG and the
PDF, whose unhinted glyphs run wider than an on-screen draw -- since a box clear at screen
resolution can still run into its neighbour in print. The fixtures are the published 27-encoder
panels (rounded to three decimals), the layouts that first put a frontier tag on a ring.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Annotation

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "bench"))

from plotting import plot_croma_pareto, plot_rank_pareto  # noqa: E402
from plotting import DEFAULT_DPI  # noqa: E402
from plotting import style as plotstyle  # noqa: E402

#: (model, median CRoMa, LTM_10, biological k-NN balanced accuracy) per tile benchmark.
CAMELYON = [
    ("RudolfV-2-S", 0.324, -0.024, 0.984),
    ("Mascaret", 0.285, -0.016, 0.981),
    ("Mettle-1.5", 0.277, 0.004, 0.988),
    ("RudolfV-2", 0.243, -0.040, 0.989),
    ("RudolfV-2-B", 0.235, -0.047, 0.987),
    ("Mettle-1 (Pre-release)", 0.204, -0.021, 0.985),
    ("Virchow2", 0.199, -0.106, 0.988),
    ("CONCH", 0.196, -0.202, 0.971),
    ("GenBio-PathFM", 0.191, -0.069, 0.983),
    ("CONCHv1.5", 0.187, -0.145, 0.971),
    ("H0-mini", 0.167, -0.160, 0.969),
    ("Virchow", 0.155, -0.181, 0.980),
    ("Phaet", 0.111, -0.177, 0.967),
    ("Midnight-12k", 0.108, -0.349, 0.976),
    ("H-optimus-1", 0.070, -0.149, 0.989),
    ("H-optimus-0", 0.052, -0.138, 0.985),
    ("UNI2-h", 0.045, -0.209, 0.986),
    ("MUSK", 0.042, -0.220, 0.958),
    ("Prov-GigaPath", 0.008, -0.191, 0.977),
    ("UNI", -0.034, -0.217, 0.982),
    ("mSTAR", -0.039, -0.239, 0.981),
    ("GPFM", -0.054, -0.290, 0.981),
    ("Hibou-B", -0.089, -0.363, 0.973),
    ("Phikon", -0.196, -0.476, 0.955),
    ("Phikon-v2", -0.208, -0.496, 0.954),
    ("Prost40M", -0.318, -0.640, 0.926),
    ("Hibou-L", -0.443, -0.659, 0.971),
]
TCGA_4X4 = [
    ("Midnight-12k", 0.396, -0.210, 0.884),
    ("Mascaret", 0.270, -0.109, 0.877),
    ("RudolfV-2-S", 0.192, -0.163, 0.849),
    ("RudolfV-2-B", 0.169, -0.137, 0.866),
    ("RudolfV-2", 0.168, -0.124, 0.876),
    ("GenBio-PathFM", 0.155, -0.193, 0.853),
    ("CONCHv1.5", 0.153, -0.130, 0.809),
    ("Mettle-1.5", 0.149, -0.093, 0.862),
    ("CONCH", 0.146, -0.152, 0.790),
    ("Virchow2", 0.128, -0.170, 0.826),
    ("H0-mini", 0.121, -0.188, 0.821),
    ("Mettle-1 (Pre-release)", 0.114, -0.112, 0.853),
    ("H-optimus-1", 0.096, -0.104, 0.883),
    ("Virchow", 0.092, -0.180, 0.788),
    ("Phaet", 0.081, -0.110, 0.786),
    ("UNI2-h", 0.075, -0.123, 0.847),
    ("mSTAR", 0.061, -0.126, 0.822),
    ("H-optimus-0", 0.061, -0.112, 0.851),
    ("MUSK", 0.051, -0.140, 0.723),
    ("GPFM", 0.048, -0.159, 0.806),
    ("UNI", 0.047, -0.120, 0.809),
    ("Prov-GigaPath", 0.047, -0.148, 0.829),
    ("Hibou-L", 0.039, -0.252, 0.760),
    ("Phikon", 0.036, -0.200, 0.794),
    ("Hibou-B", 0.036, -0.177, 0.770),
    ("Phikon-v2", 0.029, -0.187, 0.787),
    ("Prost40M", -0.010, -0.250, 0.633),
]
TOLKACH = [
    ("Midnight-12k", 0.584, -0.083, 0.975),
    ("Mascaret", 0.507, 0.009, 0.976),
    ("RudolfV-2-S", 0.485, -0.003, 0.984),
    ("CONCH", 0.439, -0.041, 0.972),
    ("Mettle-1.5", 0.426, 0.001, 0.977),
    ("RudolfV-2", 0.412, -0.011, 0.986),
    ("RudolfV-2-B", 0.407, -0.024, 0.984),
    ("CONCHv1.5", 0.392, -0.029, 0.972),
    ("GenBio-PathFM", 0.390, -0.022, 0.981),
    ("H0-mini", 0.380, -0.074, 0.967),
    ("Virchow", 0.371, -0.055, 0.970),
    ("Mettle-1 (Pre-release)", 0.367, -0.019, 0.975),
    ("Virchow2", 0.351, -0.037, 0.978),
    ("MUSK", 0.293, -0.066, 0.968),
    ("H-optimus-1", 0.260, -0.047, 0.978),
    ("Phaet", 0.249, -0.030, 0.966),
    ("H-optimus-0", 0.241, -0.080, 0.971),
    ("UNI2-h", 0.221, -0.063, 0.976),
    ("GPFM", 0.203, -0.131, 0.969),
    ("mSTAR", 0.184, -0.090, 0.971),
    ("Phikon", 0.175, -0.188, 0.961),
    ("UNI", 0.175, -0.080, 0.973),
    ("Prov-GigaPath", 0.140, -0.155, 0.959),
    ("Hibou-B", 0.134, -0.167, 0.966),
    ("Prost40M", 0.127, -0.236, 0.911),
    ("Phikon-v2", 0.122, -0.175, 0.958),
    ("Hibou-L", 0.111, -0.291, 0.956),
]
EXPOSED = {
    "camelyon": {"GPFM"},
    "tcga-4x4": {
        "GPFM",
        "GenBio-PathFM",
        "H0-mini",
        "MUSK",
        "Mascaret",
        "Midnight-12k",
        "Phaet",
        "Phikon",
        "Phikon-v2",
        "Prost40M",
        "mSTAR",
    },
    "tolkach": {"RudolfV-2", "RudolfV-2-B", "RudolfV-2-S"},
}
PANELS = {"camelyon": CAMELYON, "tcga-4x4": TCGA_4X4, "tolkach": TOLKACH}

#: (model, mean median-CRoMa rank, mean LTM_10 rank, mean biological-accuracy rank, exposed).
RANKS = [
    ("CONCH", 7.000, 14.000, 18.000, False),
    ("CONCHv1.5", 8.333, 9.667, 16.000, False),
    ("GPFM", 20.333, 19.333, 16.000, True),
    ("GenBio-PathFM", 8.000, 12.000, 7.000, True),
    ("H-optimus-0", 17.000, 10.667, 11.000, False),
    ("H-optimus-1", 14.333, 8.333, 2.667, False),
    ("H0-mini", 10.667, 16.667, 19.000, True),
    ("Hibou-B", 24.000, 21.667, 21.000, False),
    ("Hibou-L", 25.667, 27.000, 24.000, False),
    ("MUSK", 17.000, 15.667, 23.000, True),
    ("Mascaret", 2.000, 2.000, 8.333, True),
    ("Mettle-1 (Pre-release)", 10.000, 4.667, 8.333, False),
    ("Mettle-1.5", 5.333, 1.333, 5.333, False),
    ("Midnight-12k", 5.333, 22.000, 9.667, True),
    ("Phaet", 14.667, 8.667, 22.667, True),
    ("Phikon", 23.000, 24.333, 22.333, True),
    ("Phikon-v2", 25.667, 23.333, 24.333, True),
    ("Prost40M", 26.000, 26.000, 27.000, True),
    ("Prov-GigaPath", 21.333, 17.000, 17.333, False),
    ("RudolfV-2", 5.000, 6.000, 2.333, False),
    ("RudolfV-2-B", 5.333, 8.333, 4.000, False),
    ("RudolfV-2-S", 2.333, 8.000, 7.333, False),
    ("UNI", 21.000, 14.000, 13.333, False),
    ("UNI2-h", 17.000, 13.000, 8.667, False),
    ("Virchow", 12.333, 15.667, 17.667, False),
    ("Virchow2", 10.000, 12.000, 7.667, False),
    ("mSTAR", 19.333, 16.667, 14.000, True),
]


@contextmanager
def held_figure(monkeypatch):
    """The figure a plot function draws, held open past its ``plt.close``, in the house style."""
    figures: list = []
    close = plt.close

    def keep(fig=None):
        if isinstance(fig, matplotlib.figure.Figure):
            figures.append(fig)
            return None
        return close(fig)

    monkeypatch.setattr(plt, "close", keep)
    with matplotlib.rc_context():
        plotstyle.apply_style()
        yield figures


def _labels_and_disks(fig, renderer):
    """Each drawn label's extent, and each marker/ring as a disk, in ``renderer``'s pixels."""
    ax = fig.axes[0]  # the panel; the colour bar is added after it
    labels = [
        (t.get_text(), t.get_window_extent(renderer))
        for t in ax.texts
        if isinstance(t, Annotation) and t.get_text()
    ]
    disks = []
    for collection in ax.collections:
        offsets = np.asarray(collection.get_offsets())
        if offsets.size == 0:
            continue
        sizes = collection.get_sizes()
        for i, (x, y) in enumerate(ax.transData.transform(offsets)):
            radius_pt = (float(sizes[i % len(sizes)]) / np.pi) ** 0.5
            disks.append((x, y, radius_pt * fig.dpi / 72.0))
    return labels, disks


def _exported_geometry(fig, tmp_path) -> dict:
    """``_labels_and_disks`` as each export draws them, keyed by format, with the pixel size in
    points: the figure is saved exactly as ``_finalize_figure`` saves it, and the geometry is
    read inside each save's draw, in that save's renderer and at that save's dpi."""
    geometry = {}
    for fmt, kwargs in (("png", {"dpi": DEFAULT_DPI}), ("pdf", {})):

        def record(event, fmt=fmt):
            geometry[fmt] = (*_labels_and_disks(fig, event.renderer), 72.0 / fig.dpi)

        cid = fig.canvas.mpl_connect("draw_event", record)
        try:
            fig.savefig(tmp_path / f"panel.{fmt}", **kwargs)
        finally:
            fig.canvas.mpl_disconnect(cid)
    return geometry


def _covers(box, disk) -> bool:
    x, y, radius = disk
    nearest_x = min(max(x, box.x0), box.x1)
    nearest_y = min(max(y, box.y0), box.y1)
    return (nearest_x - x) ** 2 + (nearest_y - y) ** 2 < radius**2


def _collisions(fig, tmp_path) -> dict[str, list[str]]:
    """Every label-on-point and label-on-label collision, per exported format."""
    found = {}
    for fmt, (labels, disks, pt) in _exported_geometry(fig, tmp_path).items():
        found[fmt] = [
            f"{text!r} covers the point at ({x * pt:.1f}, {y * pt:.1f}) pt"
            for text, box in labels
            for x, y, radius in disks
            if _covers(box, (x, y, radius))
        ]
        found[fmt] += [
            f"{a!r} overlaps {b!r} by "
            f"{(min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)) * pt:.2f} x "
            f"{(min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)) * pt:.2f} pt"
            for i, (a, box_a) in enumerate(labels)
            for b, box_b in labels[i + 1 :]
            if box_a.overlaps(box_b)
        ]
    return found


@pytest.mark.parametrize("panel", sorted(PANELS))
def test_no_label_on_a_median_tail_panel_covers_a_point_or_another_label(
    panel, monkeypatch, tmp_path
):
    rows = [
        {"model": m, "croma": x, "croma_ltm_alpha": y, "bio": bio} for m, x, y, bio in PANELS[panel]
    ]
    with held_figure(monkeypatch) as figures:
        plot_croma_pareto(rows, tmp_path / "croma_pareto.png", exposed=EXPOSED[panel])
        (fig,) = figures
        assert _collisions(fig, tmp_path) == {"png": [], "pdf": []}


def test_no_label_on_the_rank_overview_covers_a_point_or_another_label(monkeypatch, tmp_path):
    rows = [
        {"model": m, "median_rank": x, "tail_rank": y, "bio_rank": bio, "exposed": exposed}
        for m, x, y, bio, exposed in RANKS
    ]
    with held_figure(monkeypatch) as figures:
        plot_rank_pareto(rows, tmp_path / "rank_pareto.png", n_benchmarks=3)
        (fig,) = figures
        assert _collisions(fig, tmp_path) == {"png": [], "pdf": []}
