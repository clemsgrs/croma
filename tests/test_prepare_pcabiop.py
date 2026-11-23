import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "bench", ROOT / "scripts" / "prep"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout
import prepare_pcabiop

torch = pytest.importorskip("torch")


def _write_run(run_dir: Path, vectors: dict[str, np.ndarray], *, preset: str, record: bool) -> None:
    """A PAR slide2vec run: config.yaml, per-slide .pt + .meta.json, and (new runs) the record."""
    (run_dir / "slide_embeddings").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"name": preset},
                "tiling": {"params": {"requested_spacing_um": 0.5, "requested_tile_size_px": 448}},
            }
        )
    )
    if record:
        (run_dir / "run_record.json").write_text(
            json.dumps({"slide2vec_version": "7.1.4", "slide_encoder_eval_mode": True})
        )
    for sid, vector in vectors.items():
        torch.save(torch.as_tensor(vector), run_dir / "slide_embeddings" / f"{sid}.pt")
        (run_dir / "slide_embeddings" / f"{sid}.meta.json").write_text(
            json.dumps({"sample_id": sid, "compatibility": {"encoder_name": preset}})
        )


PANDA_RUN = {
    "run_id": "titan-panda-s2v714",
    "run_dir": "/features/titan-panda-s2v714",
    "slide2vec_version": "7.1.4",
    "encoder_preset": "titan",
    "tile_size_px": 448,
    "spacing_um": 0.5,
    "aggregation_revision": None,
    "slide_encoder_eval_mode": True,
}


@pytest.fixture
def pcabiop_inputs(bench_env, tmp_path, monkeypatch):
    """Two PANDA ID slides in a three-slide panda-wsi tileset, two PAR OOD slides."""
    wsi = tmp_path / "wsi"
    (wsi / "features").mkdir(parents=True)
    panda_csv = tmp_path / "panda.csv"
    pd.DataFrame(
        {"sample_id": ["p0", "p1"], "label": [0, 1], "data_provider": ["radboud", "karolinska"]}
    ).to_csv(panda_csv, index=False)
    par_labels = wsi / "par-labels.csv"
    pd.DataFrame(
        {"sample_id": ["q0", "q1"], "biological_class": ["benign", "cancer"], "isup_r1": [0, 2]}
    ).to_csv(par_labels, index=False)

    bench_env.write_tileset(
        "panda-wsi",
        pd.DataFrame({"sample_id": ["p1", "p2", "p0"]}),
        {"TITAN": np.eye(3, dtype=np.float32), "PRISM": np.eye(3, dtype=np.float32)},
    )
    # TITAN's panda-wsi rows were rebuilt from a recorded run; PRISM's predate the record.
    layout.embedding_path("panda-wsi", "TITAN").with_suffix(".npy.json").write_text(
        json.dumps({"model_id": "TITAN", "source_runs": [{**PANDA_RUN, "n_samples": 3}]})
    )
    layout.embedding_path("panda-wsi", "PRISM").with_suffix(".npy.json").write_text(
        json.dumps({"model_id": "PRISM"})
    )
    par_vectors = {"q0": np.full(3, 0.5, np.float32), "q1": np.full(3, 0.25, np.float32)}
    _write_run(wsi / "features" / "titan-par-s2v714", par_vectors, preset="titan", record=True)
    _write_run(wsi / "features" / "552fd85i", par_vectors, preset="prism", record=False)

    monkeypatch.setattr(prepare_pcabiop, "WSI", wsi)
    monkeypatch.setattr(prepare_pcabiop, "PANDA_ID_MANIFEST", panda_csv)
    monkeypatch.setattr(prepare_pcabiop, "PAR_LABELS", par_labels)
    monkeypatch.setattr(prepare_pcabiop, "MODELS", ["TITAN", "PRISM"])
    monkeypatch.setattr(
        prepare_pcabiop, "PAR_FEATURE_DIR", {"TITAN": "titan-par-s2v714", "PRISM": "552fd85i"}
    )
    monkeypatch.setattr(prepare_pcabiop, "OOD_PER_CLASS", 1)
    monkeypatch.setattr(prepare_pcabiop, "META_PATH", tmp_path / "data" / "pcabiop.csv")
    return wsi


def _sidecar(model):
    return json.loads(layout.embedding_path("pcabiop", model).with_suffix(".npy.json").read_text())


def test_sidecar_records_the_source_run_of_each_subset(pcabiop_inputs):
    prepare_pcabiop.main()

    sidecar = _sidecar("TITAN")
    assert (sidecar["n_id"], sidecar["n_ood"], sidecar["dim"]) == (2, 2, 3)
    # ID rows are sliced from panda-wsi, so they inherit the runs that tileset recorded.
    assert sidecar["source_runs"]["ID"] == [PANDA_RUN]
    run_dir = pcabiop_inputs / "features" / "titan-par-s2v714"
    assert sidecar["source_runs"]["OOD"] == [
        {**PANDA_RUN, "run_id": "titan-par-s2v714", "run_dir": str(run_dir), "n_samples": 2}
    ]


def test_id_rows_inherit_only_the_panda_wsi_runs_that_supplied_them(pcabiop_inputs):
    # panda-wsi row p2 (not an ID slide) and row p0 (an ID slide) came from two one-slide
    # runs beside the main run; only the run behind p0 is a source of pcabiop's rows.
    def _side(run_id, sid):
        return {
            **PANDA_RUN,
            "run_id": run_id,
            "n_samples": 1,
            "sample_ids": [sid],
            "config_vs_main_run": {"main_run_id": "titan-panda-s2v714", "differences": {}},
        }

    layout.embedding_path("panda-wsi", "TITAN").with_suffix(".npy.json").write_text(
        json.dumps(
            {
                "model_id": "TITAN",
                "source_runs": [
                    {**PANDA_RUN, "n_samples": 1},
                    _side("titan-panda-p2", "p2"),
                    _side("titan-panda-p0", "p0"),
                ],
            }
        )
    )

    prepare_pcabiop.main()

    kept = _side("titan-panda-p0", "p0")
    del kept["n_samples"]
    assert _sidecar("TITAN")["source_runs"]["ID"] == [PANDA_RUN, kept]


def test_unrecorded_sources_are_left_out_or_unknown(pcabiop_inputs):
    prepare_pcabiop.main()

    runs = _sidecar("PRISM")["source_runs"]
    assert "ID" not in runs  # its panda-wsi sidecar records no source run
    (ood,) = runs["OOD"]
    assert ood["run_id"] == "552fd85i"
    assert ood["encoder_preset"] == "prism"
    assert ood["slide2vec_version"] is None
    assert ood["slide_encoder_eval_mode"] is None
