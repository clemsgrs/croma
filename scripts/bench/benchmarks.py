"""The benchmark registry: what each named benchmark actually evaluates.

A benchmark is a *view* over a tileset, not a thing that owns embeddings::

    benchmark  =  (tileset, eval manifest, evaluation design)

The eval manifest selects rows from ``output/embeddings/<tileset>/manifest.csv`` by
source row, and for ``paired_2x2`` designs may repeat a row across subsets. Two
benchmarks over the same tileset (``prostate`` and ``prostate-4class``) therefore
share one embedding matrix and cost nothing extra on disk.

Deliberately absent: a model roster. The models a benchmark evaluates are *whatever
has been embedded for its tileset* -- discovered from the ``.npy`` files present. That
is what lets a newly embedded encoder join every benchmark over that tileset without
editing this file.

The four PathoROB benchmarks read ``*-ri.csv``: the exact rows PathoROB's own
``robustness_index.get_meta`` feeds to RI. That selection is *not* uniform across
cohorts and is not derivable from a single column -- see
``scripts/prep/build_pathorob_views.py``. In particular the APD study evaluates a
different row set (the whole cohort, split by ``apd_split``), which is why RI and APD
each get their own manifest.

``k_max`` is the sweep ceiling, not the operating point. The operating point is chosen
per protocol (per-model ``k-star``, or the shared ``median-k``); see ``layout.PROTOCOLS``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Uniform ceiling of the dense k sweep (1..k_max) for every benchmark. The smallest
#: evaluated unit in the registry is 1,000 samples (panda, and panda-isup's smallest
#: subset), so a 100-neighbour ceiling fits comfortably inside every subset.
DEFAULT_K_MAX = 100


@dataclass(frozen=True)
class BenchmarkSpec:
    """One evaluable view over a tileset."""

    name: str
    tileset: str
    #: Repo-relative CSV selecting the evaluated rows (plus ``subset`` when paired).
    manifest: str
    #: ``all`` or ``paired_2x2``.
    design: str
    #: Ceiling of the dense k sweep (1..k_max).
    k_max: int = DEFAULT_K_MAX
    #: Manifest column holding the non-biological confounder.
    confounder_column: str = "medical_center"


_SPECS: tuple[BenchmarkSpec, ...] = (
    # --- PathoROB tile benchmarks -------------------------------------------------
    # Each reads its cohort's RI view; the benchmark is named for the cohort itself.
    BenchmarkSpec(
        name="pathorob-camelyon",
        tileset="pathorob-camelyon",
        manifest="data/pathorob/manifests/pathorob-camelyon-ri.csv",
        design="all",
    ),
    BenchmarkSpec(
        name="pathorob-tolkach-esca",
        tileset="pathorob-tolkach-esca",
        manifest="data/pathorob/manifests/pathorob-tolkach-esca-ri.csv",
        design="all",
    ),
    BenchmarkSpec(
        name="pathorob-tcga-2x2",
        tileset="pathorob-tcga-2x2",
        manifest="data/pathorob/manifests/pathorob-tcga-2x2-ri.csv",
        design="paired_2x2",
    ),
    BenchmarkSpec(
        name="pathorob-tcga-4x4",
        tileset="pathorob-tcga-4x4",
        manifest="data/pathorob/manifests/pathorob-tcga-4x4-ri.csv",
        design="all",
    ),
    # --- Prostate-shift views (three views, one tileset) --------------------------
    BenchmarkSpec(
        name="prostate",
        tileset="prostate-shift",
        manifest="data/prostate-shift-binary-kirumc.csv",
        design="all",
    ),
    BenchmarkSpec(
        name="prostate-4class",
        tileset="prostate-shift",
        manifest="data/prostate-shift-4class-kirumc-paired.csv",
        design="paired_2x2",
    ),
    BenchmarkSpec(
        name="prostate-gradebal",
        tileset="prostate-shift",
        manifest="data/prostate-shift-gradebal-binary-kirumc-paired.csv",
        design="paired_2x2",
    ),
    # --- PANDA slide benchmarks (two views, one tileset) --------------------------
    # PANDA's confounder is the slide's data provider (radboud / karolinska).
    BenchmarkSpec(
        name="panda",
        tileset="panda-wsi",
        manifest="data/benchmarks/panda.csv",
        design="all",
        confounder_column="data_provider",
    ),
    BenchmarkSpec(
        name="panda-isup",
        tileset="panda-wsi",
        manifest="data/benchmarks/panda-isup.csv",
        design="paired_2x2",
        confounder_column="data_provider",
    ),
)

BENCHMARKS: dict[str, BenchmarkSpec] = {spec.name: spec for spec in _SPECS}

#: Tileset -> the benchmarks that view it.
TILESETS: dict[str, tuple[str, ...]] = {
    tileset: tuple(s.name for s in _SPECS if s.tileset == tileset)
    for tileset in dict.fromkeys(s.tileset for s in _SPECS)
}


def get(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(
            f"unknown benchmark {name!r}; registered: {sorted(BENCHMARKS)}"
        ) from None
