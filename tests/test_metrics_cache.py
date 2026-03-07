
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import input_fingerprint as ifp
import metrics_cache as mcache


def test_cache_key_is_stable_for_semantically_equivalent_inputs() -> None:
    key_a = mcache.build_cache_key(
        artifact_name="mari_curve",
        model="M1",
        input_fingerprint={
            "manifest_fingerprint": "m",
            "embedding_fingerprint": "e",
            "excluded_centers_signature": "C1,C2",
        },
        params={
            "k_values": [3, 1, 3],
            "tau": 0.2,
        },
    )
    key_b = mcache.build_cache_key(
        artifact_name="mari_curve",
        model="M1",
        input_fingerprint={
            "embedding_fingerprint": "e",
            "excluded_centers_signature": "C1,C2",
            "manifest_fingerprint": "m",
        },
        params={
            "tau": 0.2,
            "k_values": [1, 3],
        },
    )
    assert key_a["key_hash"] == key_b["key_hash"]


def test_cache_key_changes_when_dependent_parameter_changes() -> None:
    key_a = mcache.build_cache_key(
        artifact_name="mari_curve",
        model="M1",
        input_fingerprint={
            "manifest_fingerprint": "m",
            "embedding_fingerprint": "e",
            "excluded_centers_signature": "",
        },
        params={"k_values": [1, 3], "tau": 0.2},
    )
    key_b = mcache.build_cache_key(
        artifact_name="mari_curve",
        model="M1",
        input_fingerprint={
            "manifest_fingerprint": "m",
            "embedding_fingerprint": "e",
            "excluded_centers_signature": "",
        },
        params={"k_values": [1, 3], "tau": 0.25},
    )
    assert key_a["key_hash"] != key_b["key_hash"]


def test_manifest_fingerprint_changes_with_content(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "medical_center": ["C1", "C2"],
            "slide_id": ["sl0", "sl1"],
        }
    )
    a = ifp.manifest_fingerprint(df)
    df2 = df.copy()
    df2.loc[1, "label"] = "A"
    b = ifp.manifest_fingerprint(df2)
    assert a != b


def test_manifest_fingerprint_ignores_patch_id() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "medical_center": ["C1", "C2"],
            "slide_id": ["sl0", "sl1"],
            "patch_id": ["p0", "p1"],
            "subset": ["pair0", "pair0"],
        }
    )
    a = ifp.manifest_fingerprint(df)
    df2 = df.copy()
    df2["patch_id"] = ["q0", "q1"]
    b = ifp.manifest_fingerprint(df2)
    assert a == b


def test_embedding_fingerprint_changes_with_sidecar_metadata(tmp_path: Path) -> None:
    emb = tmp_path / "emb.npy"
    np.save(emb, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    sidecar = emb.with_suffix(".npy.json")
    sidecar.write_text(
        '{"manifest":"a.csv","model_id":"x","extract":"cls","mixed_precision":false}\n',
        encoding="utf-8",
    )
    a = ifp.embedding_fingerprint(emb)
    sidecar.write_text(
        '{"manifest":"b.csv","model_id":"x","extract":"cls","mixed_precision":false}\n',
        encoding="utf-8",
    )
    b = ifp.embedding_fingerprint(emb)
    assert a != b
