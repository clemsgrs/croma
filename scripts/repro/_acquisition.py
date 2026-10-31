"""Per-centre acquisition details: curated scanner facts joined to counted cohorts.

``tab:acquisition`` states, for every benchmark x centre, how the tiles were acquired and
how many the evaluation uses. The two halves have different owners:

- **Curated facts** -- scanner, resolution (mpp) and magnification -- are properties of
  the source datasets that no file here records. They live in ``acquisition.csv`` beside
  this module, one row per benchmark x centre, each with a ``source`` citing the primary
  publication. A value without a source is written ``not documented``; ``load_curated``
  refuses a blank, which would otherwise read as "unknown" without saying so, and a
  documented value whose row cites nothing.
- **Counts** -- slides, patients, tiles, and the role each centre plays -- are counted
  from the rows the runs evaluate: the benchmark view (``views.load_view``) for the
  representation cohort, the APD metadata for the ID and OOD probe pools. A centre whose
  role pools hold the same slides and tiles is one row with both roles.

The tile size is read from the image files themselves, one tile per slide per centre,
through the tileset manifest: it is the only file that lists every tile on disk, the probe
pools' held-out centres included.

TCGA slide barcodes carry their own provenance (``tcga_barcodes``): the patient is the
barcode's first three fields, the sample type its fourth field's two digits (01 primary
tumour, 02 recurrent tumour, 06 metastatic) and the slide type its sixth field (DX for a
diagnostic slide).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "bench", REPO / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from paper_manifest import by_benchmark, by_prefix  # noqa: E402

CURATED_CSV = HERE / "acquisition.csv"

NOT_DOCUMENTED = "not documented"

#: The curated facts, each either sourced or ``NOT_DOCUMENTED``.
CURATED = ("scanner", "mpp", "magnification")
_COLUMNS = ("benchmark", "centre", "name", *CURATED, "patient_id", "note", "cite", "source")


def load_curated(path: Path = CURATED_CSV) -> pd.DataFrame:
    """The curated per-centre facts, validated so that every value is sourced or flagged."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(set(_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} lacks column(s) {missing}")
    for _, row in frame.iterrows():
        where = f"{path.name}: {row['benchmark']} / {row['centre']}"
        blank = [c for c in CURATED if not row[c].strip()]
        if blank:
            raise ValueError(f"{where} leaves {blank} blank; write {NOT_DOCUMENTED!r}")
        documented = [c for c in CURATED if row[c] != NOT_DOCUMENTED]
        if documented and not row["source"].strip():
            raise ValueError(f"{where} states {documented} with no source")
    twice = frame[frame.duplicated(["benchmark", "centre"], keep=False)]
    if len(twice):
        pairs = sorted(set(zip(twice["benchmark"], twice["centre"])))
        raise ValueError(f"{path.name} lists {pairs} twice")
    return frame


def _count(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


#: TCGA slide barcode: project-TSS-participant-sample+vial-portion+analyte-slide.
_BARCODE = re.compile(r"(TCGA-\w{2}-\w{4})-(\d{2})[A-Z]-\d{2}[A-Z]?-([A-Z]{2})\w*")

#: The sample-type codes the prose names (GDC sample type table).
PRIMARY, RECURRENT, METASTATIC = "01", "02", "06"


@dataclass(frozen=True)
class Barcodes:
    """What a cohort's TCGA slide barcodes say about its slides."""

    slides: int
    patients: int
    #: Diagnostic (``DX``) slides, as opposed to frozen-tissue (``TS``/``BS``) ones.
    diagnostic: int
    #: Slides and tiles per two-digit sample-type code.
    sample_slides: dict[str, int]
    sample_tiles: dict[str, int]


def tcga_barcodes(slide_tiles: Mapping[str, int]) -> Barcodes:
    """Parse every slide barcode; ``slide_tiles`` maps a slide to the tiles it contributes."""
    patients, diagnostic = set(), 0
    sample_slides: dict[str, int] = {}
    sample_tiles: dict[str, int] = {}
    for slide, tiles in slide_tiles.items():
        match = _BARCODE.fullmatch(slide)
        if match is None:
            raise ValueError(f"{slide!r} is not a TCGA slide barcode")
        patient, sample, kind = match.groups()
        patients.add(patient)
        diagnostic += kind == "DX"
        sample_slides[sample] = sample_slides.get(sample, 0) + 1
        sample_tiles[sample] = sample_tiles.get(sample, 0) + int(tiles)
    return Barcodes(
        slides=len(slide_tiles),
        patients=len(patients),
        diagnostic=diagnostic,
        sample_slides=dict(sorted(sample_slides.items())),
        sample_tiles=dict(sorted(sample_tiles.items())),
    )


def barcode_macros(prefix: str, b: Barcodes) -> list[str]:
    """The barcode counts as ``\\<prefix>...`` macros, for the prose and the table caption."""
    values = {
        "BarcodeSlides": b.slides,
        "BarcodePatients": b.patients,
        "DxSlides": b.diagnostic,
        "PrimarySlides": b.sample_slides.get(PRIMARY, 0),
        "RecurrentSlides": b.sample_slides.get(RECURRENT, 0),
        "RecurrentTiles": b.sample_tiles.get(RECURRENT, 0),
        "MetastaticSlides": b.sample_slides.get(METASTATIC, 0),
    }
    return [rf"\newcommand{{\{prefix}{name}}}{{{_count(v)}}}" for name, v in values.items()]


def _tcga_patient(slide: str) -> str:
    match = _BARCODE.fullmatch(slide)
    if match is None:
        raise ValueError(f"{slide!r} is not a TCGA slide barcode")
    return match.group(1)


def _par_patient(slide: str) -> str:
    # PAR names a slide c<patient>, or c<patient>a / c<patient>b for a patient's two slides.
    match = re.fullmatch(r"(c\d+)[ab]?", slide)
    if match is None:
        raise ValueError(f"{slide!r} is not a PAR slide name")
    return match.group(1)


#: How a ``patient_id`` rule in ``acquisition.csv`` reads a patient off a slide name. A
#: centre with no rule releases no patient identifiers, so its patients are not counted.
PATIENT_RULES = {"tcga-barcode": _tcga_patient, "par-filename": _par_patient}

ROLES = ("Representation", "ID probe", "OOD probe")
POOLED = "*"


@dataclass(frozen=True)
class Cohort:
    """One benchmark's evaluated rows, in the three frames the table counts.

    ``representation`` is the view's eval manifest (``sample_id``, ``confounder``,
    ``group_id``); ``probe`` the APD metadata (``subset``, ``slide_id``,
    ``medical_center``), or None when the benchmark has no probe pools; ``tileset`` the
    tileset manifest (``image_path``, ``confounder``, ``group_id``), or None for a
    slide-level benchmark, whose slide encoders tile the slides themselves.
    """

    benchmark: str
    name: str
    representation: pd.DataFrame
    probe: pd.DataFrame | None
    tileset: pd.DataFrame | None


@dataclass(frozen=True)
class Row:
    benchmark: str
    centre: str
    scanner: str
    tile_px: int | None
    mpp: str
    magnification: str
    slides: int
    patients: int | None
    tiles: int | None
    roles: tuple[str, ...]
    note: str
    cite: tuple[str, ...]


@dataclass(frozen=True)
class _Pool:
    slides: frozenset[str]
    tiles: int


def _role_pools(cohort: Cohort) -> dict[str, dict[str, _Pool]]:
    """Per centre, the slides and distinct tiles of every role it plays."""
    pools: dict[str, dict[str, _Pool]] = {}
    view = cohort.representation
    for centre, frame in view.groupby("confounder", sort=False):
        pools.setdefault(str(centre), {})[ROLES[0]] = _Pool(
            frozenset(frame["group_id"].astype(str)), int(frame["sample_id"].nunique())
        )
    if cohort.probe is not None:
        for (subset, centre), frame in cohort.probe.groupby(
            ["subset", "medical_center"], sort=False
        ):
            role = {"ID": ROLES[1], "OOD": ROLES[2]}[subset]
            pools.setdefault(str(centre), {})[role] = _Pool(
                frozenset(frame["slide_id"].astype(str)), len(frame)
            )
    return pools


def _pooled(cohort: Cohort) -> dict[str, dict[str, _Pool]]:
    if cohort.probe is not None:
        raise ValueError(f"{cohort.benchmark}: a pooled row cannot carry probe pools")
    view = cohort.representation
    return {
        POOLED: {
            ROLES[0]: _Pool(
                frozenset(view["group_id"].astype(str)), int(view["sample_id"].nunique())
            )
        }
    }


def tile_px(paths) -> int:
    """The one square size every tile at ``paths`` has; anything else raises."""
    from PIL import Image

    def size(path) -> tuple[int, int]:
        with Image.open(path) as image:
            return image.size

    # Reading a header is I/O-bound; a few threads hide the network filesystem's latency.
    with ThreadPoolExecutor(max_workers=3) as pool:
        sizes = set(pool.map(size, list(paths)))
    if len(sizes) != 1 or len({*next(iter(sizes))}) != 1:
        raise ValueError(f"expected one square tile size, found tile sizes {sorted(sizes)}")
    return next(iter(sizes))[0]


def _centre_tile_px(cohort: Cohort, centre: str) -> int | None:
    """Tile size at a centre, read from one tile of each of its slides."""
    if cohort.tileset is None:
        return None
    tiles = cohort.tileset
    if centre != POOLED:
        tiles = tiles[tiles["confounder"] == centre]
    first = tiles.drop_duplicates("group_id")["image_path"]
    try:
        return tile_px(first)
    except ValueError as err:
        raise ValueError(f"{cohort.benchmark} / {centre}: {err}") from None


def rows(curated: pd.DataFrame, cohorts: list[Cohort]) -> list[Row]:
    """The table's rows: per cohort, per curated centre, one row per distinct role pool."""
    out = []
    for cohort in cohorts:
        facts = curated[curated["benchmark"] == cohort.benchmark]
        pooled = list(facts["centre"]) == [POOLED]
        pools = _pooled(cohort) if pooled else _role_pools(cohort)
        uncurated = sorted(set(pools) - set(facts["centre"]))
        if uncurated:
            raise ValueError(f"{cohort.benchmark}: no curated row for {uncurated}")
        unevaluated = sorted(set(facts["centre"]) - set(pools))
        if unevaluated:
            raise ValueError(f"{cohort.benchmark}: the runs evaluate no rows for {unevaluated}")
        for _, fact in facts.iterrows():
            rule = PATIENT_RULES.get(fact["patient_id"]) if fact["patient_id"] else None
            if fact["patient_id"] and rule is None:
                raise ValueError(f"unknown patient_id rule {fact['patient_id']!r}")
            px = _centre_tile_px(cohort, fact["centre"])
            merged: dict[_Pool, list[str]] = {}
            for role in ROLES:
                pool = pools[fact["centre"]].get(role)
                if pool is not None:
                    merged.setdefault(pool, []).append(role)
            # A pooled row names how many centres it pools: ``{n} centres``.
            name = fact["name"].format(n=cohort.representation["confounder"].nunique())
            for pool, roles in merged.items():
                out.append(
                    Row(
                        benchmark=cohort.name,
                        centre=name,
                        scanner=fact["scanner"],
                        tile_px=px,
                        mpp=fact["mpp"],
                        magnification=fact["magnification"],
                        slides=len(pool.slides),
                        patients=None if rule is None else len({rule(s) for s in pool.slides}),
                        tiles=None if cohort.tileset is None else pool.tiles,
                        roles=tuple(roles),
                        note=fact["note"],
                        cite=tuple(c for c in fact["cite"].split(";") if c),
                    )
                )
    return out


#: The benchmarks the table lists, in its order: the tile benchmarks with probe pools, the
#: paired TCGA cohort, then the slide panel. Prostate-shift is outside the manuscript.
TABLE_BENCHMARKS = (
    "pathorob-camelyon",
    "pathorob-tcga-4x4",
    "pathorob-tolkach-esca",
    "pathorob-tcga-2x2",
    "panda",
    "panda-isup",
)

#: The slide panel's manifest entries carry no short name; the paper calls them PCaBiop.
_NAMES = {"panda": "PCaBiop", "panda-isup": "PCaBiop-ISUP"}

#: The cohorts whose slide names are TCGA barcodes, as macro prefixes.
TCGA_PREFIXES = ("TcgaFourByFour", "TcgaTwoByTwo")


def display_name(prefix: str) -> str:
    entry = by_prefix(prefix)
    return _NAMES.get(entry.benchmark) or entry.short_name


def _probe_metadata(benchmark: str) -> pd.DataFrame | None:
    """The APD metadata the probes run on, from the schedules ``tab:split-schedule`` lists."""
    from _split_schedule import TABLE_DATASETS
    from loaders import DATASETS

    paths = [DATASETS[k]["metadata"] for k in TABLE_DATASETS if DATASETS[k]["benchmark"] == benchmark]
    return pd.read_csv(REPO / paths[0]) if paths else None


def _tileset(benchmark: str) -> pd.DataFrame:
    import benchmarks
    import layout

    return pd.read_csv(layout.tileset_manifest(benchmarks.get(benchmark).tileset))


def load_cohorts() -> list[Cohort]:
    """Every listed benchmark's view, probe pools and tile files, as the runs read them."""
    import views

    cohorts = []
    for benchmark in TABLE_BENCHMARKS:
        entry = by_benchmark(benchmark)
        slide_level = entry.model_type == "slide-level"
        cohorts.append(
            Cohort(
                benchmark=benchmark,
                name=display_name(entry.prefix),
                representation=views.load_view(benchmark).eval_manifest,
                probe=_probe_metadata(benchmark),
                tileset=None if slide_level else _tileset(benchmark),
            )
        )
    return cohorts


def load_barcodes() -> dict[str, Barcodes]:
    """Every TCGA cohort's barcodes, over each slide its tileset embeds (all roles)."""
    out = {}
    for prefix in TCGA_PREFIXES:
        tiles = _tileset(by_prefix(prefix).benchmark)
        out[prefix] = tcga_barcodes(tiles.groupby("group_id").size().to_dict())
    return out


def acquisition_macros() -> list[str]:
    """The TCGA barcode macros, for ``generated_values.tex``."""
    lines = ["% TCGA slide provenance, from the barcodes of every embedded slide"]
    for prefix, b in load_barcodes().items():
        lines += barcode_macros(prefix, b)
    return lines
