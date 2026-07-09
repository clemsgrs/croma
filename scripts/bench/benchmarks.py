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

``k_max`` is the sweep ceiling, not the operating point. The operating point is chosen
per protocol (per-model ``k-star``, or the shared ``median-k``); see ``layout.PROTOCOLS``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """One evaluable view over a tileset."""

    name: str
    tileset: str
    #: Repo-relative CSV selecting the evaluated rows (plus ``subset`` when paired).
    manifest: str
    #: ``dataset_wide`` or ``paired_2x2``.
    design: str
    #: Ceiling of the dense k sweep (1..k_max).
    k_max: int
    #: Manifest column holding the non-biological confounder.
    confounder_column: str = "medical_center"


_SPECS: tuple[BenchmarkSpec, ...] = (
    # --- PathoROB tile benchmarks -------------------------------------------------
    BenchmarkSpec(
        name="camelyon",
        tileset="pathorob-camelyon",
        manifest="data/pathorob/manifests/pathorob-camelyon-faithful.csv",
        design="dataset_wide",
        k_max=100,
    ),
    BenchmarkSpec(
        name="tolkach",
        tileset="pathorob-tolkach-esca",
        manifest="data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv",
        design="dataset_wide",
        k_max=100,
    ),
    BenchmarkSpec(
        name="tcga-2x2",
        tileset="pathorob-tcga-2x2",
        manifest="data/pathorob/manifests/pathorob-tcga-2x2.csv",
        design="paired_2x2",
        k_max=100,
    ),
    # The unfiltered views: every centre, not just PathoROB's in-domain pair. The
    # pretraining-overlap table reads tolkach-full's per-sample metrics.
    BenchmarkSpec(
        name="camelyon-full",
        tileset="pathorob-camelyon",
        manifest="data/pathorob/manifests/pathorob-camelyon.csv",
        design="dataset_wide",
        k_max=25,
    ),
    BenchmarkSpec(
        name="tolkach-full",
        tileset="pathorob-tolkach-esca",
        manifest="data/pathorob/manifests/pathorob-tolkach-esca.csv",
        design="dataset_wide",
        k_max=25,
    ),
    BenchmarkSpec(
        name="tcga-4x4",
        tileset="pathorob-tcga-4x4",
        manifest="data/pathorob/manifests/pathorob-tcga-4x4.csv",
        design="dataset_wide",
        k_max=100,
    ),
    # --- Prostate-shift views (three views, one tileset) --------------------------
    BenchmarkSpec(
        name="prostate",
        tileset="prostate-shift",
        manifest="data/prostate-shift-binary-kirumc.csv",
        design="dataset_wide",
        k_max=25,
    ),
    BenchmarkSpec(
        name="prostate-4class",
        tileset="prostate-shift",
        manifest="data/prostate-shift-4class-kirumc-paired.csv",
        design="paired_2x2",
        k_max=25,
    ),
    BenchmarkSpec(
        name="prostate-gradebal",
        tileset="prostate-shift",
        manifest="data/prostate-shift-gradebal-binary-kirumc-paired.csv",
        design="paired_2x2",
        k_max=25,
    ),
    # --- PANDA slide benchmarks (two views, one tileset) --------------------------
    # PANDA's confounder is the slide's data provider (radboud / karolinska). These
    # manifests had no home under data/; migration writes them to data/benchmarks/.
    BenchmarkSpec(
        name="panda",
        tileset="panda-wsi",
        manifest="data/benchmarks/panda.csv",
        design="dataset_wide",
        k_max=25,
        confounder_column="data_provider",
    ),
    # The stale k-star artifact for this benchmark recorded k_max=500, but it was written
    # by an older build: it disagrees with the same code's later median-k run (which used
    # 25) on croma_auc/croma_delta over identical data. 25 matches `panda` and the most
    # recent actual run; k* here is 13, well inside either ceiling.
    BenchmarkSpec(
        name="panda-isup",
        tileset="panda-wsi",
        manifest="data/benchmarks/panda-isup.csv",
        design="paired_2x2",
        k_max=25,
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
