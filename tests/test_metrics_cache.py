import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
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
        },
        params={
            "evaluation_design": "all",
            "k_values": [3, 1, 3],
            "tau": 0.2,
        },
    )
    key_b = mcache.build_cache_key(
        artifact_name="mari_curve",
        model="M1",
        input_fingerprint={
            "embedding_fingerprint": "e",
            "manifest_fingerprint": "m",
        },
        params={
            "tau": 0.2,
            "k_values": [1, 3],
            "evaluation_design": "all",
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
        },
        params={"evaluation_design": "all", "k_values": [1, 3], "tau": 0.2},
    )
    key_b = mcache.build_cache_key(
        artifact_name="mari_curve",
        model="M1",
        input_fingerprint={
            "manifest_fingerprint": "m",
            "embedding_fingerprint": "e",
        },
        params={"evaluation_design": "all", "k_values": [1, 3], "tau": 0.25},
    )
    assert key_a["key_hash"] != key_b["key_hash"]


def test_manifest_fingerprint_changes_with_content(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "confounder": ["VendorA", "VendorB"],
            "group_id": ["sl0", "sl1"],
        }
    )
    df.attrs["confounder_column"] = "scanner_vendor"
    a = ifp.manifest_fingerprint(df)
    df2 = df.copy()
    df2.loc[1, "label"] = "A"
    df2.attrs["confounder_column"] = "scanner_vendor"
    b = ifp.manifest_fingerprint(df2)
    assert a != b


def test_manifest_fingerprint_changes_with_selected_confounder_column(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "confounder": ["VendorA", "VendorB"],
            "group_id": ["sl0", "sl1"],
        }
    )
    df.attrs["confounder_column"] = "scanner_vendor"
    a = ifp.manifest_fingerprint(df)
    df.attrs["confounder_column"] = "scanner_batch"
    b = ifp.manifest_fingerprint(df)
    assert a != b


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
