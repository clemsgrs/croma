import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_embeddings as ee
import model_registry as mr


def test_registry_includes_conch_and_midnight_models() -> None:
    registry = mr._build_model_registry()

    assert "CONCH" in registry
    assert registry["CONCH"].backend == "conch_v1"
    assert registry["CONCH"].extract == "raw"

    assert "CONCHv1.5" in registry
    assert registry["CONCHv1.5"].backend == "conch_v1_5"
    assert registry["CONCHv1.5"].extract == "raw"
    assert registry["CONCHv1.5"].mixed_precision is True

    assert "Midnight-12k" in registry
    assert registry["Midnight-12k"].backend == "midnight"
    assert registry["Midnight-12k"].model_id == "kaiko-ai/midnight"
    assert registry["Midnight-12k"].extract == "cls_and_patch"


def test_registry_includes_ported_slide2vec_models() -> None:
    registry = mr._build_model_registry()

    assert registry["mSTAR"].backend == "timm"
    assert registry["mSTAR"].model_id == "hf-hub:Wangyh/mSTAR"
    assert registry["mSTAR"].extract == "cls"

    assert registry["DINOv2-B"].backend == "timm"
    assert registry["DINOv2-B"].model_id == "vit_base_patch14_dinov2.lvd142m"
    assert registry["DINOv2-B"].extract == "cls"

    assert registry["GPFM"].backend == "gpfm"
    assert registry["GPFM"].model_id == "majiabo/GPFM"

    assert registry["MUSK"].backend == "musk"
    assert registry["MUSK"].model_id == "hf_hub:xiangjx/musk"
    assert registry["MUSK"].extract == "ms_aug"
    assert registry["MUSK"].mixed_precision is True

    assert registry["GenBio-PathFM"].backend == "genbio"
    assert registry["GenBio-PathFM"].model_id == "genbio-ai/genbio-pathfm"


def test_parse_models_rejects_empty_and_duplicates() -> None:
    with pytest.raises(ValueError, match="empty"):
        mr._parse_models("UNI,,Phikon")
    with pytest.raises(ValueError, match="duplicate"):
        mr._parse_models("UNI,UNI")


def test_unwrap_gpfm_state_dict_handles_wrappers_and_prefixes() -> None:
    bare = {"cls_token": 1, "blocks.0.norm.weight": 2}
    assert ee._unwrap_gpfm_state_dict(dict(bare)) == bare

    wrapped = {"teacher": {"module.cls_token": 1, "backbone.blocks.0.w": 2}}
    assert ee._unwrap_gpfm_state_dict(wrapped) == {"cls_token": 1, "blocks.0.w": 2}

    nested = {"state_dict": {"backbone.pos_embed": 3}}
    assert ee._unwrap_gpfm_state_dict(nested) == {"pos_embed": 3}


def test_unwrap_gpfm_state_dict_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="state_dict mapping"):
        ee._unwrap_gpfm_state_dict(["not", "a", "state", "dict"])
