"""The manifest's independence key is ``group_id``.

Every canonical manifest carries a ``group_id``: the identifier of the independence
group a sample belongs to (a slide, a patient, a specimen, an acquisition -- whatever
unit the study declares non-independent). Candidates sharing a query's ``group_id`` are
never eligible neighbours. These tests pin that contract at the seams that expose it:
the manifest loader, the metric computations, the alignment key, and the fingerprint.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa, RI
from croma.alignment import build_embedding_source_manifest
from croma.metrics.pairs import load_manifest, normalize_manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from input_fingerprint import manifest_fingerprint  # noqa: E402

CONFOUNDER_COLUMN = "scanner_vendor"


def _unit(angle_deg: float) -> list[float]:
    angle = math.radians(angle_deg)
    return [math.cos(angle), math.sin(angle)]


#: One query plus the neighbours whose types are known by construction.
#:
#: Row 0 is the query ``q`` (label ``A``, confounder ``V1``). Row 1 is an ``OS``
#: impostor (wrong label, same confounder) sitting almost on top of it; row 2 is a far
#: ``OS`` impostor; row 3 is the query's nearest ``SO`` ally (same label, other
#: confounder). Row 4 completes the 2x2 so the same rows can be scored paired.
_ANGLES = (0.0, 5.0, 120.0, 60.0, 130.0)
_LABELS = ("A", "B", "B", "A", "B")
_CONFOUNDERS = ("V1", "V1", "V1", "V2", "V2")


def _neighbourhood_manifest(*, impostor_group: str) -> pd.DataFrame:
    """The five rows above, with the near impostor's group under the caller's control."""
    groups = ["g0", impostor_group, "g2", "g3", "g4"]
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(_ANGLES))],
            "image_path": [f"/tmp/{i}.png" for i in range(len(_ANGLES))],
            "label": list(_LABELS),
            CONFOUNDER_COLUMN: list(_CONFOUNDERS),
            "group_id": groups,
            "subset": ["p1"] * len(_ANGLES),
            "dataset": ["toy"] * len(_ANGLES),
        }
    )


def _neighbourhood_features() -> np.ndarray:
    return np.asarray([_unit(a) for a in _ANGLES], dtype=float)


def _query_croma(manifest: pd.DataFrame, *, evaluation_design: str) -> float:
    result = CRoMa.compute(
        _neighbourhood_features(),
        manifest,
        confounder_column=CONFOUNDER_COLUMN,
        evaluation_design=evaluation_design,
        m=1,
        start_k=1,
    )
    return float(np.asarray(result.sample_values, dtype=float)[0])


def test_manifest_with_group_id_is_accepted_and_scored() -> None:
    manifest = _neighbourhood_manifest(impostor_group="g1")

    croma_q = _query_croma(manifest, evaluation_design="all")

    # Nearest OS impostor is ~0.004 away, nearest SO ally 0.5: confounder-dominant.
    assert croma_q < 0.0


@pytest.mark.parametrize("evaluation_design", ["all", "paired_2x2"])
def test_same_group_candidate_is_excluded_even_when_nearest(evaluation_design: str) -> None:
    with_distinct_groups = _query_croma(
        _neighbourhood_manifest(impostor_group="g1"),
        evaluation_design=evaluation_design,
    )
    with_shared_group = _query_croma(
        _neighbourhood_manifest(impostor_group="g0"),
        evaluation_design=evaluation_design,
    )

    # Sharing the query's group makes the nearest impostor ineligible; the next one
    # is farther than the ally, so the margin flips from confounder- to biology-dominant.
    assert with_distinct_groups < 0.0
    assert with_shared_group > 0.0


def test_manifest_with_only_slide_id_names_the_missing_group_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    manifest_path.write_text(
        "sample_id,image_path,label,scanner_vendor,slide_id\n"
        "s0,/tmp/0.png,A,V1,sl0\n"
        "s1,/tmp/1.png,B,V2,sl1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_manifest(str(manifest_path), confounder_column=CONFOUNDER_COLUMN)

    message = str(excinfo.value)
    assert "group_id" in message
    # The manifest carries the pre-rename name, so say so: there is no alias to fall
    # back on, and the fix is a rename the caller has to make.
    assert "slide_id" in message


def test_slide_id_alongside_group_id_is_ordinary_metadata() -> None:
    manifest = _neighbourhood_manifest(impostor_group="g0")
    manifest["slide_id"] = ["sl0", "sl1", "sl2", "sl3", "sl4"]

    # ``slide_id`` disagrees with ``group_id`` on row 1; only ``group_id`` may be read.
    assert _query_croma(manifest, evaluation_design="all") > 0.0


@pytest.mark.parametrize("blank", ["", "   ", None, float("nan")])
def test_blank_or_missing_group_id_fails_clearly(blank: object) -> None:
    manifest = _neighbourhood_manifest(impostor_group="g1")
    manifest.loc[2, "group_id"] = blank

    with pytest.raises(ValueError, match="group_id"):
        normalize_manifest(
            manifest,
            confounder_column=CONFOUNDER_COLUMN,
            source="manifest",
        )


def test_blank_group_id_is_reported_by_row_position_on_any_index() -> None:
    # A caller who filtered a manifest hands over a frame whose index is neither
    # contiguous nor integer; the row it names must still be the row it means.
    manifest = _neighbourhood_manifest(impostor_group="g1")
    manifest.index = [f"r{i}" for i in range(len(manifest))]
    manifest.loc["r2", "group_id"] = ""

    with pytest.raises(ValueError, match=r"group_id values at rows \[2\]"):
        normalize_manifest(
            manifest,
            confounder_column=CONFOUNDER_COLUMN,
            source="manifest",
        )


def test_insufficient_eligible_neighbours_is_reported_group_neutrally() -> None:
    manifest = _neighbourhood_manifest(impostor_group="g0")
    manifest["group_id"] = "one-group"

    with pytest.raises(RuntimeError) as excinfo:
        RI.compute(
            _neighbourhood_features(),
            manifest,
            confounder_column=CONFOUNDER_COLUMN,
            evaluation_design="all",
            k_candidates=[1],
        )

    message = str(excinfo.value)
    assert "group" in message
    assert "slide" not in message


def test_group_id_is_part_of_the_embedding_alignment_key() -> None:
    manifest = _neighbourhood_manifest(impostor_group="g1")
    other = manifest.copy()
    other["group_id"] = [f"other-{g}" for g in manifest["group_id"]]
    stacked = pd.concat([manifest, other], ignore_index=True)
    stacked = stacked.rename(columns={CONFOUNDER_COLUMN: "confounder"})

    embedding_manifest, row_to_source = build_embedding_source_manifest(stacked)

    assert "group_id" in embedding_manifest.columns
    assert len(embedding_manifest) == len(stacked)
    assert row_to_source.tolist() == list(range(len(stacked)))


def test_group_id_changes_the_manifest_fingerprint() -> None:
    manifest = _neighbourhood_manifest(impostor_group="g1").rename(
        columns={CONFOUNDER_COLUMN: "confounder"}
    )
    changed = manifest.copy()
    changed.loc[0, "group_id"] = "g0-renamed"

    assert manifest_fingerprint(manifest) != manifest_fingerprint(changed)
