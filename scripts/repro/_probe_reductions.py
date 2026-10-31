"""Shared basis for ``tab:low-data``, the sensitivity and specificity rows of
``tab:nipd-auroc``, and their macros.

The study (``scripts/studies/probe_reductions.py``) writes its CSVs and ``study.json`` under
``output/studies/probe-reductions/``; ``study.json`` records the sha256 of every file the
study read and of each CSV it wrote. :func:`load` refuses a study whose recorded inputs or
outputs no longer hash to those values -- a regenerated record, a re-run APD cell or
benchmark -- with ``CaptionClaimError``, and a study that was never run with
``FileNotFoundError``: nothing is rendered from a stale or partial study, and nothing is
skipped.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import probe_reductions as study  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

STUDY_REL = study.STUDY_REL
_RERUN = "re-run scripts/studies/probe_reductions.py"


@dataclass(frozen=True)
class ProbeReductions:
    """The study's tables, verified current."""

    info: dict
    coverage: pd.DataFrame
    sens_spec: pd.DataFrame
    low_data: pd.DataFrame
    correlations: pd.DataFrame

    @property
    def low_data_cohorts(self) -> list[str]:
        return list(self.info["low_data_cohorts"])

    def tile_levels(self, cohort: str) -> list[int]:
        """``cohort``'s tile levels, full first."""
        return [int(t) for t in self.info["tile_levels"][cohort]]

    @property
    def sens_spec_correlations(self) -> pd.DataFrame:
        return self.correlations[self.correlations["analysis"] == "sens_spec"]

    @property
    def low_data_correlations(self) -> pd.DataFrame:
        return self.correlations[self.correlations["analysis"] == "low_data"]

    @property
    def n_ranked(self) -> int:
        """The ranked encoders behind every low-data number: one panel on every cohort."""
        counts = (
            self.low_data[~self.low_data["control"]].groupby(["cohort", "tiles"])["model"].count()
        )
        if counts.nunique() != 1:
            raise CaptionClaimError(f"the low-data cohorts rank different panels: {dict(counts)}")
        return int(counts.iloc[0])


def _verify(study_dir: Path, info: dict, root: Path) -> None:
    for name in study.OUTPUTS:
        path = study_dir / name
        if not path.is_file():
            raise CaptionClaimError(f"{path}: missing study output; {_RERUN}")
        if study._sha256(path) != info["outputs"][name]:
            raise CaptionClaimError(f"{path}: changed since the study wrote it; {_RERUN}")
    for relative, expected in info["inputs"].items():
        path = Path(relative) if Path(relative).is_absolute() else root / relative
        if not path.is_file():
            raise CaptionClaimError(f"{relative}: study input missing; {_RERUN}")
        if study._sha256(path) != expected:
            raise CaptionClaimError(f"{relative}: changed since the study read it; {_RERUN}")


def _csv(path: Path) -> pd.DataFrame:
    # ``undefined`` is empty where no encoder dropped out; keep it a string column.
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "undefined" in frame:
        frame["undefined"] = frame["undefined"].fillna("").astype(str)
    return frame


def load(root: Path = REPO) -> ProbeReductions:
    """Read the study after checking every recorded fingerprint.

    Raises:
        FileNotFoundError: If the study has not been run.
        CaptionClaimError: If an input or output changed since the study ran.
    """
    root = Path(root)
    study_dir = root / STUDY_REL
    info_path = study_dir / "study.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{info_path} is absent; run scripts/studies/probe_reductions.py")
    info = json.loads(info_path.read_text("utf-8"))
    _verify(study_dir, info, root)
    return ProbeReductions(
        info=info,
        coverage=_csv(study_dir / "coverage.csv"),
        sens_spec=_csv(study_dir / "sens_spec.csv"),
        low_data=_csv(study_dir / "low_data.csv"),
        correlations=_csv(study_dir / "correlations.csv"),
    )
