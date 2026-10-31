"""Shared basis for the sample-shortcut floats and macros (``tab:sample-shortcut*``).

The study (``scripts/studies/sample_shortcut.py``) writes one directory per cohort under
``output/studies/sample-shortcut/``: ``encoders.csv``, ``samples.csv.gz`` and
``cohort.json``, which records the sha256 of every file the study read and of the two it
wrote. :func:`load` refuses a study whose recorded inputs or outputs no longer hash to those
values -- a re-run benchmark, a regenerated record, new embeddings -- with
``CaptionClaimError``, and a study that was never run with ``FileNotFoundError``: the
tables and macros are never rendered from a stale or partial study, and never skipped.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "studies" / "apd", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _paper_tables import CaptionClaimError, croma_as_margin  # noqa: E402
from loaders import DATASETS  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402
from plotting.style import CONTROL_MODEL, published_models  # noqa: E402

#: Where the study writes, relative to the checkout.
STUDY_REL = "output/studies/sample-shortcut"

#: The cohorts in the order the supplement presents them; the first two carry colour.
COHORTS = ("camelyon", "tcga_4x4", "pcabiop")
TILE_COHORTS = ("camelyon", "tcga_4x4")

#: Study outputs whose sha256 ``cohort.json`` records.
OUTPUTS = ("encoders.csv", "samples.csv.gz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CohortResult:
    """One cohort's study, verified current, with encoders in the run's table order."""

    cohort: str
    info: dict
    #: Pathology encoders, by the run's median CRoMa (descending); published names.
    ranked: pd.DataFrame
    #: The natural-image control, if the run has it; excluded from every summary.
    control: pd.DataFrame
    study_dir: Path

    @property
    def run(self):
        return by_benchmark(self.info["benchmark"])

    @property
    def colour(self) -> bool:
        return "colour_balanced_accuracy" in self.info

    def samples(self) -> pd.DataFrame:
        """Every encoder's held-out rows (registry names, as the study wrote them)."""
        return pd.read_csv(
            self.study_dir / "samples.csv.gz",
            dtype={"model": str, "sample_id": str, "group_id": str, "image_path": str},
        )


def _verify(study_dir: Path, info: dict, root: Path) -> None:
    for name in OUTPUTS:
        path = study_dir / name
        if not path.is_file():
            raise CaptionClaimError(f"{path}: missing study output")
        if _sha256(path) != info["outputs"][name]:
            raise CaptionClaimError(f"{path}: changed since the study wrote it; re-run the study")
    for relative, expected in info["inputs"].items():
        path = Path(relative) if Path(relative).is_absolute() else root / relative
        if not path.is_file():
            raise CaptionClaimError(f"{relative}: study input missing; re-run the study")
        if _sha256(path) != expected:
            raise CaptionClaimError(
                f"{relative}: changed since the study read it; re-run "
                "scripts/studies/sample_shortcut.py"
            )


def summary(result: CohortResult) -> dict:
    """The cross-encoder numbers the text, captions and macros quote for one cohort.

    AUROC medians and interval counts are over the *evaluable* pathology encoders (at least
    ``min_flipped`` flips; the control never counts). ``n_ci_above_half`` counts encoders
    whose CRoMa AUROC interval lies wholly above 0.5, ``n_above_bio`` those whose paired
    interval for CRoMa - biology-only margin lies wholly above 0 (``below`` the mirror
    images). The colour correlation median is over every pathology encoder.
    """
    ranked = result.ranked
    evaluable = ranked[ranked["evaluable"].astype(bool)]
    out = {
        "n_ranked": int(len(ranked)),
        "n_evaluable": int(len(evaluable)),
        "min_flipped": int(result.info["min_flipped"]),
        "n_tiles": int(result.info["n_tiles"]),
        "n_held_out": int(result.info["n_held_out"]),
        "n_never_held_out": int(result.info["n_never_held_out"]),
        "flipped_median": float(ranked["n_flipped"].median()),
        "croma_auroc_median": float(evaluable["auroc_croma"].median()),
        "bio_auroc_median": float(evaluable["auroc_bio"].median()),
        "purity_auroc_median": float(evaluable["auroc_purity"].median()),
        "n_ci_above_half": int((evaluable["auroc_croma_lo"] > 0.5).sum()),
        "n_ci_below_half": int((evaluable["auroc_croma_hi"] < 0.5).sum()),
    }
    for baseline in ("bio", "purity"):
        key = f"diff_croma_{baseline}"
        out[f"n_above_{baseline}"] = int((evaluable[f"{key}_lo"] > 0).sum())
        out[f"n_below_{baseline}"] = int((evaluable[f"{key}_hi"] < 0).sum())
    if result.colour:
        out.update(
            colour_balanced_accuracy=float(result.info["colour_balanced_accuracy"]),
            colour_auroc_median=float(evaluable["auroc_colour"].median()),
            croma_given_colour_median=float(evaluable["auroc_croma_given_colour"].median()),
            n_given_colour_above_half=int((evaluable["auroc_croma_given_colour_lo"] > 0.5).sum()),
            rho_colour_median=float(ranked["rho_croma_colour"].median()),
        )
    return out


def load(cohort: str, root: Path = REPO) -> CohortResult:
    """Read one cohort's study after checking every recorded fingerprint.

    Raises:
        FileNotFoundError: If the study has not been run for ``cohort``.
        CaptionClaimError: If an input or output changed since the study ran, or the study's
            roster is not the run's.
    """
    root = Path(root)
    study_dir = root / STUDY_REL / cohort
    info_path = study_dir / "cohort.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{info_path} absent: run scripts/studies/sample_shortcut.py --cohorts {cohort}"
        )
    info = json.loads(info_path.read_text())
    if info.get("benchmark") != DATASETS[cohort]["benchmark"]:
        raise CaptionClaimError(f"{info_path} describes {info.get('benchmark')!r}")
    _verify(study_dir, info, root)

    run = by_benchmark(info["benchmark"])
    metrics = pd.read_csv(root / run.metrics_rel, keep_default_na=False)
    encoders = pd.read_csv(study_dir / "encoders.csv")
    if set(encoders["model"]) != set(metrics["model"]) or len(encoders) != len(metrics):
        raise CaptionClaimError(
            f"{cohort}: the study's encoders are not the run's roster "
            f"(study-only {sorted(set(encoders['model']) - set(metrics['model']))}, "
            f"run-only {sorted(set(metrics['model']) - set(encoders['model']))})"
        )
    median = dict(zip(metrics["model"], croma_as_margin(metrics["croma"])))
    encoders = encoders.assign(registry_model=encoders["model"], croma_median=encoders["model"].map(median))
    is_control = encoders["model"] == CONTROL_MODEL
    order = ["croma_median", "model"]
    ranked = encoders[~is_control].sort_values(order, ascending=[False, True], kind="stable")
    return CohortResult(
        cohort=cohort,
        info=info,
        ranked=published_models(ranked).reset_index(drop=True),
        control=published_models(encoders[is_control]).reset_index(drop=True),
        study_dir=study_dir,
    )


#: Tiles per montage row: the lowest-CRoMa row and the highest-CRoMa row.
MONTAGE_TILES = 8


@dataclass(frozen=True)
class MontageChoice:
    """Which encoder, cell and tiles ``fig:margin-montage`` shows for one cohort."""

    cohort: str
    model: str
    registry_model: str
    auroc: float
    cell: str
    n_flipped_in_cell: int
    n_in_cell: int
    #: The ``n`` lowest- and highest-CRoMa tiles, each in ascending CRoMa order.
    low: pd.DataFrame
    high: pd.DataFrame


def montage_choice(result: CohortResult, *, n: int = MONTAGE_TILES) -> MontageChoice:
    """The montage's encoder, cell and tiles, chosen by a rule fixed before looking.

    The encoder is the evaluable pathology encoder at the panel's *lower* median
    within-cell CRoMa AUROC (sorted by AUROC, then name), so the montage shows a typical
    encoder rather than the best. The cell is that encoder's class x centre cell holding the
    most flipped tiles (ties by cell name), and the tiles are the ``n`` lowest- and ``n``
    highest-CRoMa tiles among the cell's tiles the balanced probe never got wrong -- the
    population the AUROC is computed over -- ties broken by sample id.
    """
    evaluable = result.ranked[result.ranked["evaluable"].astype(bool)]
    if evaluable.empty:
        raise CaptionClaimError(f"{result.cohort}: no evaluable encoder to show")
    ordered = evaluable.sort_values(["auroc_croma", "model"], kind="stable").reset_index(drop=True)
    pick = ordered.iloc[(len(ordered) - 1) // 2]
    samples = result.samples()
    rows = samples[(samples["model"] == pick["registry_model"]) & (samples["e0"] == 0)]
    flips = rows.groupby("cell")["flipped"].sum().reset_index()
    flips = flips.sort_values(["flipped", "cell"], ascending=[False, True], kind="stable")
    cell = str(flips["cell"].iloc[0])
    in_cell = rows[rows["cell"] == cell].sort_values(["croma_m5", "sample_id"], kind="stable")
    return MontageChoice(
        cohort=result.cohort,
        model=str(pick["model"]),
        registry_model=str(pick["registry_model"]),
        auroc=float(pick["auroc_croma"]),
        cell=cell,
        n_flipped_in_cell=int(flips["flipped"].iloc[0]),
        n_in_cell=int(len(in_cell)),
        low=in_cell.head(n).reset_index(drop=True),
        high=in_cell.tail(n).reset_index(drop=True),
    )
