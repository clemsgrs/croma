import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "bench", ROOT / "scripts" / "prep"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import slide2vec_provenance as prov


def _write_run(
    run_dir: Path,
    *,
    preset: str = "gigapath-slide",
    tile_size_px: int = 224,
    spacing_um: float = 0.5,
    aggregation_revisions: dict[str, int | None] | None = None,
) -> list[Path]:
    """A slide2vec run directory as 7.1.4 leaves it: config.yaml + per-slide sidecars."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"name": preset, "allow_non_recommended_settings": False},
                "tiling": {
                    "params": {
                        "requested_spacing_um": spacing_um,
                        "requested_tile_size_px": tile_size_px,
                    }
                },
            }
        )
    )
    revisions = aggregation_revisions if aggregation_revisions is not None else {"s0": 1, "s1": 1}
    paths = []
    for sid, revision in revisions.items():
        compatibility = {"encoder_name": preset}
        if revision is not None:
            compatibility["aggregation_revision"] = revision
        (run_dir / "slide_embeddings").mkdir(exist_ok=True)
        (run_dir / "slide_embeddings" / f"{sid}.meta.json").write_text(
            json.dumps({"sample_id": sid, "compatibility": compatibility})
        )
        paths.append(run_dir / "slide_embeddings" / f"{sid}.pt")
    return paths


def test_recorded_run_provenance_comes_from_config_record_and_slide_sidecars(tmp_path):
    run_dir = tmp_path / "features" / "gigapath-par-s2v714"
    paths = _write_run(run_dir, preset="gigapath-slide", tile_size_px=224, spacing_um=0.5)
    prov.record_run(run_dir, slide2vec_version="7.1.4", slide_encoder_eval_mode=True)

    assert prov.run_provenance(run_dir, paths) == {
        "run_id": "gigapath-par-s2v714",
        "run_dir": str(run_dir),
        "slide2vec_version": "7.1.4",
        "encoder_preset": "gigapath-slide",
        "tile_size_px": 224,
        "spacing_um": 0.5,
        "aggregation_revision": 1,
        "slide_encoder_eval_mode": True,
    }


def test_a_run_without_a_record_leaves_version_and_mode_unknown(tmp_path):
    # Runs launched before the record existed (the published PRISM/MOOZY PAR runs) still
    # say what config.yaml says; what they never logged stays None rather than guessed.
    run_dir = tmp_path / "features" / "552fd85i"
    paths = _write_run(
        run_dir, preset="prism", tile_size_px=224, aggregation_revisions={"s0": None}
    )

    record = prov.run_provenance(run_dir, paths)

    assert record["encoder_preset"] == "prism"
    assert record["tile_size_px"] == 224
    assert record["aggregation_revision"] is None
    assert record["slide2vec_version"] is None
    assert record["slide_encoder_eval_mode"] is None


def test_slides_of_one_run_must_agree_on_the_aggregation_revision(tmp_path):
    # A run resumed across an aggregation change would mix two position encodings.
    run_dir = tmp_path / "features" / "mixed"
    paths = _write_run(run_dir, aggregation_revisions={"s0": 1, "s1": None})
    prov.record_run(run_dir, slide2vec_version="7.1.4", slide_encoder_eval_mode=True)

    with pytest.raises(ValueError, match="aggregation_revision"):
        prov.run_provenance(run_dir, paths)


def test_a_run_cannot_be_resumed_under_another_slide2vec_version(tmp_path):
    run_dir = tmp_path / "features" / "titan-panda-s2v714"
    prov.record_run(run_dir, slide2vec_version="7.1.4", slide_encoder_eval_mode=True)
    prov.record_run(run_dir, slide2vec_version="7.1.4", slide_encoder_eval_mode=True)

    with pytest.raises(ValueError, match="7.1.4"):
        prov.record_run(run_dir, slide2vec_version="7.1.5", slide_encoder_eval_mode=True)
    assert json.loads((run_dir / prov.RUN_RECORD_NAME).read_text())["slide2vec_version"] == "7.1.4"


def test_eval_mode_means_no_submodule_of_the_loaded_encoder_is_training():
    torch = pytest.importorskip("torch")

    class _Encoder:  # a slide2vec encoder wraps its torch module(s) as attributes
        def __init__(self):
            self._model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1)).eval()
            self._device = "cpu"

    encoder = _Encoder()
    assert prov.slide_encoder_eval_mode(encoder) is True

    # Prov-GigaPath's create_model returns the module in training mode; one active
    # dropout anywhere in it is enough to make the embeddings irreproducible.
    encoder._model[1].train()
    assert prov.slide_encoder_eval_mode(encoder) is False
