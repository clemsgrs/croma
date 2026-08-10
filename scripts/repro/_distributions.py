"""Shared logic for the per-sample CRoMa distribution floats.

Camelyon backs the main-text ridgeline (results.tex Section 3.3); TCGA-4x4 and Tolkach-ESCA
back two supplementary ridgelines, and PANDA backs a slide-level ridgeline. Each tile panel
draws the full encoder roster plus the natural-image control; the slide panel has no such
control because DINOv2-B produces no whole-slide embedding. The per-model tail statistics live
here so the figure and float generators share one source of truth. Mirrors ``_apd.py`` and
``_cross_benchmark.py``.

Every run directory is resolved through ``paper_manifest`` (never a spelled-out protocol), so
the tail analysis always reads whatever protocol the paper reports for that benchmark.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _paper_tables import (  # noqa: E402
    CROMA_HEADLINE_M,
    CaptionClaimError,
    croma_as_margin,
)
from _model_provenance import exposed_models_for_domain  # noqa: E402
from paper_manifest import ResultsTable, by_prefix  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench")
)  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import CONTROL_MODEL, published_models  # noqa: E402

#: The Camelyon run backs the main-text tail analysis; its protocol comes from the manifest.
CAMELYON = by_prefix("Camelyon")

#: The two additional tile benchmarks whose distributions the supplement shows, in paper order.
SUPP_BENCHMARKS: tuple[ResultsTable, ...] = (by_prefix("TcgaFourByFour"), by_prefix("Tolkach"))

#: The slide-level PANDA run backs the supplement's fourth ridgeline (supp:panda). Its panel
#: is four whole-slide encoders and no natural-image control, so it is kept apart from the tile
#: SUPP_BENCHMARKS rather than folded into them.
PANDA = by_prefix("Panda")

#: model_metadata.csv, the same provenance table the cross-benchmark rank figure reads.
METADATA = HERE.parent / "bench" / "model_metadata.csv"


def has_exposure_domain(entry: ResultsTable) -> bool:
    """Whether a scored cohort has a declared corpus or institutional provenance domain."""
    return bool(entry.exposure_domain)


@dataclass(frozen=True)
class Model:
    """One encoder's pooled median and lower-tail summary on the tail-analysis run."""

    name: str
    median: float  # pooled CRoMa (margin scale)
    ltm: float  # LTM_10, the mean of the worst-decile margins
    f0: float  # F(0), the fraction of samples with per-sample CRoMa < 0, in [0, 1]
    is_control: bool


@dataclass(frozen=True)
class Distributions:
    """The tail-analysis run, as every model ranked best-median-first."""

    models: tuple[Model, ...]

    @property
    def by_name(self) -> dict[str, Model]:
        return {m.name: m for m in self.models}

    @property
    def pathology(self) -> tuple[Model, ...]:
        return tuple(m for m in self.models if not m.is_control)

    @property
    def control(self) -> Model:
        (ctrl,) = [m for m in self.models if m.is_control]
        return ctrl


def load(root: Path | None = None, entry: ResultsTable = CAMELYON) -> Distributions:
    """Load per-model median / LTM_10 / F(0) for a benchmark's tail-analysis run.

    Defaults to Camelyon (the main-text ridgeline); pass ``entry`` for one of the
    ``SUPP_BENCHMARKS``. Medians and LTM come from ``metrics.csv`` (the same columns the
    results table prints); F(0) is the fraction of defined per-sample margins at or below
    zero, read from ``per_sample_metrics.csv`` at the headline radius -- the column the
    freshness fixtures also write.
    """
    root = Path(root) if root is not None else HERE.parents[1]
    # The run stores registry identities; everything downstream of this loader is a
    # published surface (figures, floats, captions), so the restyle happens here.
    metrics = published_models(pd.read_csv(root / entry.metrics_rel))
    per_sample = published_models(pd.read_csv(root / entry.per_sample_rel))
    return build(metrics, per_sample)


def _f0_closed(s: pd.Series) -> float:
    """F(0) over the defined occurrences, with the boundary closed.

    An exact zero is confounder-dominant and non-finite values leave the denominator --
    the same definition ``CRoMaResult.f0`` computes inside the library.
    """
    defined = s[np.isfinite(s)]
    return float((defined <= 0.0).mean())


def build(metrics: pd.DataFrame, per_sample: pd.DataFrame) -> Distributions:
    """Assemble the ranked model list from the two frames (pure; the fixtures drive it)."""
    median = croma_as_margin(metrics["croma"])
    # F(0) on the bounded margin: normalise the per-sample column before counting, so a run
    # stored as the raw distance ratio (neutral at 1, never negative) is not read as 0% fragile.
    sample_col = f"croma_m{int(CROMA_HEADLINE_M)}"
    per_sample = per_sample.assign(_margin=croma_as_margin(per_sample[sample_col]))
    f0 = per_sample.groupby("model")["_margin"].apply(_f0_closed)

    models = [
        Model(
            name=str(row["model"]),
            median=float(median.iloc[i]),
            ltm=float(row["croma_ltm_alpha"]),
            f0=float(f0.get(str(row["model"]), float("nan"))),
            is_control=str(row["model"]) == CONTROL_MODEL,
        )
        for i, (_, row) in enumerate(metrics.iterrows())
    ]
    models.sort(key=lambda m: (m.median, m.name), reverse=True)
    return Distributions(models=tuple(models))


def assert_control_is_the_control(dist: Distributions) -> Model:
    """The control is the natural-image control, so the caption may name it so."""
    ctrl = dist.by_name.get(CONTROL_MODEL)
    if ctrl is None or not ctrl.is_control:
        raise CaptionClaimError(
            f"the caption names {CONTROL_MODEL} as the natural-image control, but it "
            f"is not in the roster, or is no longer flagged as the control."
        )
    return ctrl


def metadata_exposed_models(entry: ResultsTable, roster: set[str]) -> frozenset[str]:
    """Resolve corpus/institutional exposure from the single model metadata source."""
    # Rosters reaching this point carry published names; restyle the metadata to match.
    md = published_models(pd.read_csv(METADATA))
    return exposed_models_for_domain(md, entry.exposure_domain, roster)


def exposed_models(entry: ResultsTable, dist: Distributions) -> frozenset[str]:
    """Encoders whose pretraining overlaps a cohort in ``entry``'s benchmark, over the panel's
    ranked roster (the control is never marked).

    The Pareto figure marks exactly this set (a dagger after each exposed encoder's name in the
    legend) and its caption counts it, so the two read one source. TCGA benchmarks reuse the
    ``corpus_domains`` tags the rank-aggregate overview also reads. CAMELYON exposure is read
    from the same field; conservative Charit\'e/CHA exposure is read from
    ``institutional_domains``. A benchmark with no declared domain returns an empty set, so the
    figure simply draws no marks.
    """
    roster = {m.name for m in dist.pathology}
    return metadata_exposed_models(entry, roster)


# --- slide-level (PANDA) predicates: the supplement's fourth ridgeline (supp:panda) ---------


def assert_no_natural_image_control(dist: Distributions) -> None:
    """The slide panel carries no natural-image control, so its caption counts only encoders.

    DINOv2-B produces no whole-slide embedding, so it never enters a slide run; the slide
    caption says "all N whole-slide encoders" with no control band -- unlike the tile captions,
    which name the control. If a control ever appears the figure would draw a ridgeline the
    caption does not account for, so raise rather than ship the mismatch.
    """
    controls = [m.name for m in dist.models if m.is_control]
    if controls:
        raise CaptionClaimError(
            f"the slide distribution caption counts only whole-slide encoders, but the run "
            f"carries the natural-image control {controls}. Name it in the caption (as the "
            f"tile captions do), or drop it from the slide panel."
        )
