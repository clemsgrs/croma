"""Shared data loaders and configuration for the APD experiment trio.

``apd_experiment.py`` computes APD, ``apd_croma_correlation.py`` correlates it with the
faithful CRoMa/RI/MaRI metrics, and ``apd_figure.py`` plots it. They share the repo
root, the per-dataset APD configuration, and the data loaders collected here, so the
paths and join logic live in exactly one place.

Nothing here computes a metric. The probe protocol and both reductions ship in
``croma.downstream`` (ADR-0011), and the schedules PathoROB authored come from
``croma.downstream.pathorob_schedule``; what this module owns is which rows a cohort
consists of, how they are indexed, and the two schedules croma authored for the datasets
PathoROB has none for. There is no PathoROB checkout in the picture -- its dataset files
arrive under ``data/pathorob/`` like every other input, and its code arrives vendored,
inside the installed package.
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Repo root autodetected from this file's location: scripts/studies/apd/loaders.py
# -> parents[3] is the croma repo root, so the reproduction works from any checkout.
REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import layout  # noqa: E402  (on-disk output layout: output/embeddings/<tileset>/...)
from croma.downstream import pathorob_schedule  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402

#: The tile-level PathoROB views that make one coherent robustness/downstream panel.
#: TCGA-2x2 has no APD split and therefore cannot enter this study; prostate-shift and
#: PCaBiop are separate studies and are deliberately outside the expanded tile panel.
PATHOROB_DOWNSTREAM_DATASETS = ("camelyon", "tcga_4x4", "tolkach")

#: Fixed panel cardinality after issue #132: 25 pathology encoders plus one natural-image
#: floor. Model identity and relationships themselves remain owned by model_metadata.csv.
PATHOROB_PATHOLOGY_MODELS = 25
PATHOROB_PANEL_MODELS = PATHOROB_PATHOLOGY_MODELS + 1


def pathorob_tile_panel(metadata_path: Path | None = None) -> pd.DataFrame:
    """Return the ordered PathoROB tile panel and its model relationships.

    The study must not treat every stray ``.npy`` in an embedding directory as a
    study participant. The model metadata is already the single source of truth for panel
    identity, ordering, controls, and parent/student or parent/fine-tuned relationships;
    this narrow view makes those facts part of the downstream summary too.
    """
    path = metadata_path or REPO / "scripts" / "bench" / "model_metadata.csv"
    metadata = pd.read_csv(path)
    panel = (
        metadata.loc[
            metadata["panel"] == "tile",
            ["model", "panel_order", "parent_model", "variant_role"],
        ]
        .sort_values("panel_order")
        .reset_index(drop=True)
    )
    if panel["model"].duplicated().any():
        duplicates = sorted(panel.loc[panel["model"].duplicated(False), "model"].unique())
        raise ValueError(f"tile model metadata contains duplicate models: {duplicates}")
    if len(panel) != PATHOROB_PANEL_MODELS:
        raise ValueError(
            f"expanded PathoROB panel must contain {PATHOROB_PANEL_MODELS} models, "
            f"got {len(panel)} in {path}"
        )
    controls = panel["model"] == CONTROL_MODEL
    if controls.sum() != 1 or int((~controls).sum()) != PATHOROB_PATHOLOGY_MODELS:
        raise ValueError(
            "expanded PathoROB panel must contain 25 pathology encoders and exactly "
            f"one {CONTROL_MODEL} control"
        )
    panel["ranked"] = ~controls
    return panel[["model", "parent_model", "variant_role", "ranked"]]


#: Column naming the independence unit in the APD metadata CSVs. croma's canonical
#: manifests call this ``group_id``, but the APD metadata is PathoROB's own published
#: file -- joined against the source manifests on ``(slide_id, patch_id)`` -- and
#: renaming a source-dataset column is out of scope for the manifest contract.
APD_METADATA_GROUP_COLUMN = "slide_id"

# ---------------------------------------------------------------------------
# APD computation config (used by apd_experiment.py)
# ---------------------------------------------------------------------------
# Per-dataset constants, copied from PathoROB's apd/utils.load_data so that our
# (centre, biology) cell indexing matches the split schedule exactly. ``split_key``
# is the name get_patches_map_to_split expects (note: tcga_4x4 -> "tcga"). ``src`` is
# the *tileset* whose full embedding matrix (every row, ID + OOD centres) APD reads --
# resolved via layout.embeddings_dir(); it is not a benchmark eval view. ``benchmark``
# is the paper_manifest key naming the *run* whose CRoMa/RI/MaRI this APD is correlated
# against; it coincides with ``src`` for the PathoROB tilesets but not for prostate,
# where one tileset ("prostate-shift") backs a run named "prostate". ``centers_id`` orders
# the confounder axis of every schedule, so it is load-bearing; ``centers_ood`` records
# which centres the held-out arm holds, but the partition itself is the metadata's
# ``subset`` column, which is what ``load_data`` reads.
DATASETS = {
    "camelyon": dict(
        src="pathorob-camelyon",
        benchmark="pathorob-camelyon",
        metadata="data/pathorob/metadata/camelyon.csv",
        split_key="camelyon",
        centers_id=["RUMC", "UMCU"],
        centers_ood=["CWZ", "RST", "LPON"],
        biological_classes=["normal", "tumor"],
        num_splits=8, num_slides_per_category=17, num_patches_per_slide=300,
    ),
    "tcga_4x4": dict(
        src="pathorob-tcga-4x4",
        benchmark="pathorob-tcga-4x4",
        metadata="data/pathorob/metadata/tcga_4x4.csv",
        split_key="tcga",
        centers_id=["Asterand", "Christiana Healthcare", "Roswell Park", "University of Pittsburgh"],
        centers_ood=["Cureline", "Greater Poland Cancer Center", "International Genomics Consortium", "Johns Hopkins"],
        biological_classes=["Breast_invasive_carcinoma", "Colon_adenocarcinoma", "Lung_adenocarcinoma", "Lung_squamous_cell_carcinoma"],
        num_splits=7, num_slides_per_category=12, num_patches_per_slide=30,
    ),
    # tolkach_esca is the one cohort whose slides cannot be ordered freely: a case there
    # contributes patches to several biological classes, so it may not be trained on in one
    # and tested on in another. PathoROB enumerates the case splits that avoid this, and
    # each replicate draws one and pushes its held-out cases to the tail (see
    # ``case_arrangement``). ``case_splits`` is that enumeration -- PathoROB's own
    # ``pathorob/resources/tolkach_splits.json``, copied under ``data/pathorob/`` exactly as
    # its metadata CSVs are, because it describes the dataset rather than the protocol.
    "tolkach": dict(
        src="pathorob-tolkach-esca",
        benchmark="pathorob-tolkach-esca",
        metadata="data/pathorob/metadata/tolkach_esca.csv",
        case_splits="data/pathorob/tolkach_splits.json",
        split_key="tolkach_esca",
        centers_id=["VALSET2_WNS", "VALSET4_CHA_FULL"],
        centers_ood=["VALSET1_UKK", "VALSET3_TCGA"],
        biological_classes=["TUMOR", "MUSC_PROP", "SH_OES", "SH_MAG", "REGR_TU", "ADVENT"],
        num_splits=4, num_slides_per_category=9, num_patches_per_slide=100,
    ),
    # prostate-shift-binary: a 2-centre x 2-class design, structurally like Camelyon, but
    # with a croma-authored split schedule (split_key="prostate", see _prostate_split_map)
    # instead of Camelyon's. ID = the mirrored KI/RUMC pair (1,850 patches/cell = 50 slides
    # x 37 patches, uniform), held out OOD = NUS. Camelyon's schedule reserves a fixed 15
    # slides/cell for train+val and was tuned to its 17-slide cells, leaving only ~2 test
    # slides/cell -- a known weakness (test estimate rests on 2 slides per fold). With 50
    # slides/cell we instead use balanced load M=16, confounder step Delta=2, num_splits=9:
    # train 64*npps=2,368 (conserved across splits), test = 50-(2M+1) = 17 slides/cell
    # (idxTest=33*npps=1,221 << 1,850), reaching full confounding (favorable cell -> 0) at
    # the last split since M is even. num_patches_per_slide governs only ID pseudo-slide
    # chunking; KI/RUMC are genuinely 37/slide so pseudo-slides == real slides (slide-
    # disjoint train/test). NUS is a flat OOD test pool (no npps chunking), so its ragged
    # 24-79 patches/slide is irrelevant. See paper for the schedule-faithfulness note.
    "prostate": dict(
        src="prostate-shift",
        benchmark="prostate",
        metadata="data/prostate/metadata/prostate_shift.csv",
        split_key="prostate",
        centers_id=["KI", "RUMC"],
        centers_ood=["NUS"],
        biological_classes=["benign", "tumor"],
        num_splits=9, num_slides_per_category=50, num_patches_per_slide=37,
    ),
    # pcabiop: slide-level APD. ID = PANDA's cancer-detection cohort (the published
    # `panda` benchmark: 250 benign + 250 cancer per provider, a balanced 2x2 with 250
    # slides/cell); held-out OOD = PAR, an external Leica-scanned prostate-biopsy cohort
    # (162 benign / 162 cancer). One slide embedding per case, so num_patches_per_slide=1
    # and pseudo-slides == real slides. The schedule is croma-authored (_pcabiop_split_map),
    # anchored on Camelyon (the binary 2x2 PathoROB benchmark). Built by
    # scripts/prep/prepare_pcabiop.py.
    "pcabiop": dict(
        src="pcabiop",
        benchmark="panda",
        metadata="data/pcabiop/metadata/pcabiop.csv",
        split_key="pcabiop",
        centers_id=["radboud", "karolinska"],
        centers_ood=["PAR"],
        biological_classes=["benign", "cancer"],
        num_splits=11, num_slides_per_category=250, num_patches_per_slide=1,
    ),
}


def _prostate_split_map(split, num_patches_per_slide):
    """croma-authored split schedule for prostate-shift-binary.

    Mirrors PathoROB's Camelyon schedule structure (2 centres x 2 classes, a
    centre<->class spurious correlation injected by skewing the per-cell training
    counts) but is re-parameterised for prostate's 50 slides/cell instead of
    Camelyon's 17. Camelyon hard-codes a balanced load of 7 slides/cell, unit
    confounder step, and reserves 2*14+1 = 15 of its 17 slides for train+val --
    leaving only ~2 test slides/cell. Re-using that here would idle 35 of our 50
    slides in test. We instead use:

        balanced load   M     = 16 slides/cell   (split 0: every cell = 16*npps)
        confounder step Delta = 2 slides/split
        num_splits            = M/Delta + 1 = 9  (split 8: favourable cell -> 0)

    so favourable cells carry (M - Delta*split)*npps training patches and
    unfavourable cells (M + Delta*split)*npps. max_train_slides = 2M = 32, hence
    idxTest = 33*npps = 1,221 and a fixed 50-33 = 17-slide (629-patch) test set
    per cell; total train = 64*npps = 2,368, conserved across splits. M is even so
    Delta=2 still reaches the fully-confounded endpoint (favourable cell = 0),
    matching Camelyon's maximal-confounder split.

    PathoROB's own get_patches_map_to_split is left untouched/verbatim; this is the
    single croma-authored deviation, documented as such in the paper.
    """
    M, delta = 16, 2
    favourable = (M - delta * split) * num_patches_per_slide
    unfavourable = (M + delta * split) * num_patches_per_slide
    tss0_pairs = [(0, 0, favourable), (0, 1, unfavourable)]
    tss1_pairs = [(1, 0, unfavourable), (1, 1, favourable)]
    return sorted(tss0_pairs + tss1_pairs), 2 * M


def _pcabiop_split_map(split, num_patches_per_slide):
    """croma-authored slide-level split schedule for the PCaBiop APD experiment.

    Same 2-centre x 2-class diagonal structure as Camelyon's schedule (PathoROB), the
    binary 2x2 benchmark, re-parameterised for PANDA's 250 slides/cell:

        balanced load   M     = 100 slides/cell
        confounder step Delta = 10 slides/split
        num_splits            = M/Delta + 1 = 11  (split 10: favourable cell -> 0)

    max_train_slides = 2M = 200 (80% of the cell, matching Camelyon's 14/17 = 82%); total
    train = 4M = 400 slides, conserved across splits. num_patches_per_slide is 1 (one
    embedding per slide), so favourable/unfavourable counts are in slides directly.
    Camelyon uses Delta=1 only because its cells hold 7 slides; the faithful invariant is
    the number of points sampled on the Cramer's-V axis (Camelyon = 8), and 11 here just
    samples the same 0->1 axis more densely. The two slide-level deviations from PathoROB
    -- this schedule and the validation fraction in apd_experiment.compute -- are
    documented as such in the paper.
    """
    M, delta = 100, 10
    favourable = (M - delta * split) * num_patches_per_slide
    unfavourable = (M + delta * split) * num_patches_per_slide
    tss0_pairs = [(0, 0, favourable), (0, 1, unfavourable)]
    tss1_pairs = [(1, 0, unfavourable), (1, 1, favourable)]
    return sorted(tss0_pairs + tss1_pairs), 2 * M


#: The schedules croma authored, for the two datasets PathoROB has none for. Everything
#: else routes to its own helper, in the package.
CROMA_SPLIT_MAPS = {"prostate": _prostate_split_map, "pcabiop": _pcabiop_split_map}


def split_schedule(dataset, num_patches_per_slide=1):
    """The schedule the probe sweep walks for ``dataset``: one split per entry.

    Each entry is the ``(split_map, max_train_slides)`` pair ``croma.probe_sweep`` reads --
    how many training rows every (centre, class) cell contributes at that split, balanced
    first and fully confounded last. PathoROB owns all three of its own schedules and they
    come from the package, built by its vendored helper; prostate and pcabiop are
    croma-authored above and are built here.
    """
    cfg = DATASETS[dataset]
    split_map = CROMA_SPLIT_MAPS.get(cfg["split_key"])
    if split_map is not None:
        return [split_map(s, num_patches_per_slide) for s in range(cfg["num_splits"])]
    return pathorob_schedule(
        cfg["split_key"],
        rows_per_slide=num_patches_per_slide,
        n_splits=cfg["num_splits"],
    )


def cramers_v(table):
    """Cramer's V of a contingency table: 0 = independent, 1 = one row fixes the column.

    Scale-invariant, so slide counts and patch counts give the same answer.
    """
    t = np.asarray(table, dtype=float)
    n = t.sum()
    expected = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / n
    chi2 = ((t - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def training_correlations(dataset):
    """Cramer's V of the (centre x biological class) *training-count* table, per split.

    The schedule injects the spurious centre<->biology correlation by skewing how many
    training patches each (centre, class) cell contributes. This is that skew, summarised in
    one number per split, and it is what the per-split probe accuracies are plotted against.

    PathoROB's own figures index the same schedule by the favourable cell's slide fraction
    (7/14 ... 14/14 on Camelyon). That is monotone in V, but its lower endpoint moves with
    the number of classes, so the three benchmarks cannot share one axis. V can: every
    schedule starts balanced at 0 and ends fully confounded at 1.
    """
    cfg = DATASETS[dataset]
    shape = (len(cfg["centers_id"]), len(cfg["biological_classes"]))
    correlations = []
    for split_map, _ in split_schedule(dataset):
        table = np.zeros(shape)
        for center_idx, bio_idx, num_patches in split_map:
            table[center_idx, bio_idx] = num_patches
        correlations.append(cramers_v(table))
    return correlations


class Cohort(NamedTuple):
    """One (model, dataset) cohort, indexed the way ``croma.probe_sweep`` reads it.

    ``embeddings`` holds the ID rows in metadata order, and ``confounders``/``labels`` give
    each of them its (centre, biological class) cell as *indices into the schedule* -- the
    schedule addresses cells by position, so the names are mapped here, once. The OOD rows
    ride along as a test set the sweep scores but never trains on. ``group_ids`` is what
    Tolkach's case arrangement reads; it is the ID rows' own column, in the same order.

    The independence unit is spelled ``group_id`` in croma's canonical manifests but read
    here out of the APD metadata's ``slide_id``: those CSVs are PathoROB's published files,
    joined against the source manifests on ``(slide_id, patch_id)``, and renaming a
    source-dataset column is out of scope for the manifest contract.
    """

    embeddings: np.ndarray
    confounders: np.ndarray
    labels: np.ndarray
    ood_embeddings: np.ndarray
    ood_labels: np.ndarray
    group_ids: list


def load_data(model, dataset):
    """Assemble one cohort out of croma's embedding cache and PathoROB's metadata.

    This is the study's half of the arrangement ADR-0011 draws: the library consumes
    embeddings and a split assignment and knows nothing about where a repository keeps its
    files, so somebody has to read the manifest, pick the model's matrix out of the cache
    and map centre and class names onto the schedule's indices. That is all this does.

    The row order is the metadata's, which is what makes the cache usable at all: croma's
    per-tileset embedding matrix is row-aligned with PathoROB's metadata CSV, so grouping
    rows into cells in that order reproduces PathoROB's own per-cell pseudo-slide chunking
    bit for bit.
    """
    cfg = DATASETS[dataset]
    centers_id = cfg["centers_id"]
    bio = cfg["biological_classes"]

    md = pd.read_csv(REPO / cfg["metadata"])
    emb = np.load(layout.embeddings_dir(cfg["src"]) / f"{model}.npy")
    assert len(emb) == len(md), f"{model}/{dataset}: {len(emb)} emb vs {len(md)} meta rows"

    md_id = md[md["subset"] == "ID"]
    md_ood = md[md["subset"] == "OOD"]

    return Cohort(
        embeddings=emb[md_id.index.to_numpy()],
        confounders=np.array([centers_id.index(c) for c in md_id["medical_center"]]),
        labels=np.array([bio.index(b) for b in md_id["biological_class"]]),
        ood_embeddings=emb[md_ood.index.to_numpy()],
        ood_labels=np.array([bio.index(b) for b in md_ood["biological_class"]]),
        group_ids=list(md_id[APD_METADATA_GROUP_COLUMN]),
    )


def arrangement_for(dataset, group_ids):
    """How ``dataset`` orders each cell's slides per replicate, or None for the default.

    Only tolkach_esca needs one, and it needs it because of how its cases are annotated
    rather than because of anything about the protocol -- which is why the sweep takes it
    as an argument and this study supplies it.
    """
    cfg = DATASETS[dataset]
    if "case_splits" not in cfg:
        return None
    path = REPO / cfg["case_splits"]
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset}: {path} is missing. It is PathoROB's own "
            "pathorob/resources/tolkach_splits.json -- copy it there, as the metadata CSVs "
            "beside it were copied."
        )
    return case_arrangement(cfg["centers_id"], json.loads(path.read_text()), group_ids)


def case_arrangement(centers, feasible_splits, group_ids):
    """Order a replicate's slides so a drawn set of cases falls in the held-out tail.

    Tolkach-ESCA's cases span biological classes, so a case trained on in one class and
    tested on in another leaks. PathoROB enumerates the case splits that avoid it; each
    replicate draws one per centre and moves those cases' slides to the back of every cell,
    which is where the sweep's held-out rows come from.

    Returned as the ``arrange_slides`` hook ``croma.downstream.probe_sweep_over_test_sets``
    takes, so this is the only per-replicate logic the study still runs -- and it runs
    inside the library's sweep rather than beside a copy of it. The draws come first and
    the shuffle second, off the one generator, because that is the order the sweep's own
    arrangement and PathoROB's driver both consume it in; swapping them would re-seed every
    Tolkach number.

    The intersection below is ``sorted``, and that is a deliberate divergence from PathoROB's
    driver, which leaves it a bare ``set``. Iteration order over a set of strings depends on
    hashing, which Python randomises per process, and the slides land in the held-out tail in
    the order they are iterated -- so under the bare set it is hash-dependent *which* cases sit
    outermost. The tail is narrow (one slide past the widest training block, two slides per
    cell here) and a replicate draws two to five cases, so more cases are pushed back than the
    tail holds and the surplus stops just short of it. That makes the test set's membership,
    not merely its order, a function of ``PYTHONHASHSEED``.

    Sorting costs bit-parity with PathoROB on this cohort: every Tolkach number moves. The
    trade was taken knowingly, on the repository owner's instruction, because the reference
    behaviour it preserved was itself unreproducible -- the pre-rewire driver had the same
    property, so the stored Tolkach matrices were never bit-reproducible either. See #105.
    """

    def arrange(cells, rng):
        drawn = tuple(rng.randint(0, len(feasible_splits["train"][center]) - 1) for center in centers)
        for i, center in enumerate(centers):
            test_cases = feasible_splits["test"][center][drawn[i]]
            for slides in cells[i]:
                rng.shuffle(slides)
                cases = [group_ids[slide[0]] for slide in slides]
                for test_case in sorted(set(test_cases) & set(cases)):
                    at = [group_ids[slide[0]] for slide in slides].index(test_case)
                    slides.append(slides.pop(at))
        return cells

    return arrange


# ---------------------------------------------------------------------------
# APD <-> metric correlation config + loaders (used by apd_croma_correlation.py
# and apd_figure.py)
# ---------------------------------------------------------------------------
# The three tile-level benchmarks reported in the downstream-correlation *table*. PCaBiop is
# kept out of that table because four encoders is too few for a rank correlation to carry a
# conclusion -- but "not tabulated" is not "not computed": supp/panda.tex quotes its two
# Spearman values in prose, under an explicit "these associations are descriptive" caveat.
# So the exclusion lives in the table generator (_apd.FIGURE_DATASETS), not here.
# Prostate-shift remains available to the experiment driver but is not part of the
# manuscript's active benchmark panel.
DATASET_KEYS = ["camelyon", "tcga_4x4", "tolkach"]
#: Datasets the *join* covers, which is deliberately wider than the ones the correlation
#: table reports. PCaBiop is excluded from the table (four encoders is too few for a rank
#: correlation) but its figure is still rendered and cited, and that figure's scatter needs
#: CRoMa joined onto its APD rows. Driving the join off ``DATASET_KEYS`` silently dropped
#: those four rows, leaving ``apd_figure.py --pcabiop`` to fail with an empty scatter roster
#: -- excluded from a *conclusion* is not the same as excluded from the *data*.
#: Prostate-shift stays out: the experiment driver computes it, but nothing joins or plots it.
JOIN_KEYS = [*DATASET_KEYS, "pcabiop"]
#: Everything this study reads and writes: the APD CSVs, the join, and the plots drawn from
#: them. Figures land here beside their data rather than under ``paper/`` -- see the note on
#: ``OUT`` in ``scripts/repro/figures/apd_figure.py``.
STUDY_DIR = REPO / "output/studies/apd"
CORR_METRICS = ["croma", "ri", "mari"]

def metric_csv(dataset):
    """The metrics.csv of the run the paper reports for ``dataset``'s benchmark.

    APD itself is protocol-free (it trains probes on embeddings, never on a k-NN graph), but
    the metrics it is correlated *against* are not: RI and MaRI are k-dependent, so a
    Spearman(RI, APD) must be computed at the protocol whose RI the paper prints. CRoMa is
    k-free, so its row of tab:apd-correlation is protocol-invariant. This used to name
    ``k-star/pathorob-*`` -- runs archived when the tile panel moved to ``median-k``, so the
    join could no longer run at all. See ADR-0010 and CONTEXT.md ("Study").

    The manifest is imported here rather than at module scope because it is paper tooling,
    absent from a clone (ADR-0012), and only the join needs it: computing APD must not
    depend on the manuscript's build.
    """
    from paper_manifest import by_benchmark

    return by_benchmark(DATASETS[dataset]["benchmark"]).metrics_rel


def ranked(df):
    """Drop the natural-image control: it is a floor, not a competitor, so it holds no
    rank and enters no Spearman. Every cross-model statistic over APD goes through here,
    so the figure's in-panel rho cannot drift from the table's."""
    return df[df["model"] != CONTROL_MODEL]


def load_joined(apd_csv):
    """Join APD results with the faithful CRoMa/RI/MaRI metrics, per (dataset, model).

    The control is kept in the join so its APD stays on record; every consumer that ranks
    or plots models drops it through ``ranked``.
    """
    apd = pd.read_csv(apd_csv)
    frames = []
    for ds in JOIN_KEYS:
        m = pd.read_csv(REPO / metric_csv(ds))
        m["dataset"] = ds  # align with APD's dataset key (CSV stores the dir name)
        # Defensive: the canonical dirs are already signed-margin CRoMa; only convert
        # if a legacy ratio-scale CSV (any value > 1) is ever pointed at here.
        if m["croma"].max() > 1.0:
            m["croma"] = (m["croma"] - 1.0) / (m["croma"] + 1.0)
        frames.append(m)
    faith = pd.concat(frames, ignore_index=True)
    df = apd.merge(faith[["dataset", "model", *CORR_METRICS, "croma_ltm_alpha"]], on=["dataset", "model"], how="inner")
    missing = set(zip(apd["dataset"], apd["model"])) - set(zip(df["dataset"], df["model"]))
    if missing:
        print(f"[warn] {len(missing)} (dataset,model) APD rows had no faithful metric and were dropped: {sorted(missing)}")
    return df


def read_joined():
    """Load the APD<->metric join written by ``apd_croma_correlation.main``.

    Written by ``apd_croma_correlation.py`` with CRoMa already on the signed-margin
    scale (the paper's definition).
    """
    return pd.read_csv(STUDY_DIR / "apd_metrics_joined.csv")
