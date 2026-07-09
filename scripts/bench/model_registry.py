import dataclasses


@dataclasses.dataclass
class ModelSpec:
    backend: str  # "timm", "hf_auto", "conch_v1", "conch_v1_5", "midnight", "gpfm", "musk", or "genbio"
    model_id: str
    extract: str = "cls"
    timm_kwargs: dict = dataclasses.field(default_factory=dict)
    mixed_precision: bool = False


def _parse_models(raw_models: str) -> list[str]:
    models = [m.strip() for m in str(raw_models).split(",")]
    if any(not m for m in models):
        raise ValueError("Invalid --models value: empty model name detected.")
    if len(set(models)) != len(models):
        raise ValueError("Invalid --models value: duplicate model names detected.")
    return models


def _build_model_registry():
    virchow_kwargs = {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}
    uni2h_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": "SwiGLUPacked",
        "act_layer": "SiLU",
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    h0_mini_kwargs = {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}

    return {
        "Virchow2": ModelSpec(
            backend="timm",
            model_id="hf-hub:paige-ai/Virchow2",
            extract="virchow",
            timm_kwargs=virchow_kwargs,
            mixed_precision=True,
        ),
        "Virchow": ModelSpec(
            backend="timm",
            model_id="hf-hub:paige-ai/Virchow",
            extract="cls_and_patch",
            timm_kwargs=virchow_kwargs,
            mixed_precision=True,
        ),
        "UNI2-h": ModelSpec(
            backend="timm",
            model_id="hf-hub:MahmoodLab/UNI2-h",
            extract="cls",
            timm_kwargs=uni2h_kwargs,
        ),
        "UNI": ModelSpec(
            backend="timm",
            model_id="hf-hub:MahmoodLab/uni",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
        ),
        "CONCHv1.5": ModelSpec(
            backend="conch_v1_5",
            model_id="MahmoodLab/conch-v1.5",
            extract="raw",
            mixed_precision=True,
        ),
        "CONCH": ModelSpec(
            backend="conch_v1",
            model_id="MahmoodLab/conch",
            extract="raw",
        ),
        "H-optimus-1": ModelSpec(
            backend="timm",
            model_id="hf-hub:bioptimus/H-optimus-1",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
            mixed_precision=True,
        ),
        "H-optimus-0": ModelSpec(
            backend="timm",
            model_id="hf-hub:bioptimus/H-optimus-0",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
            mixed_precision=True,
        ),
        "H0-mini": ModelSpec(
            backend="timm",
            model_id="hf-hub:bioptimus/H0-mini",
            extract="virchow",
            timm_kwargs=h0_mini_kwargs,
            mixed_precision=True,
        ),
        "Prov-GigaPath": ModelSpec(
            backend="timm",
            model_id="hf-hub:prov-gigapath/prov-gigapath",
            extract="cls",
        ),
        "Midnight-12k": ModelSpec(
            backend="midnight",
            model_id="kaiko-ai/midnight",
            extract="cls_and_patch",
        ),
        "Prost40M": ModelSpec(
            backend="timm",
            model_id="hf-hub:waticlems/Prost40M",
            extract="cls",
            timm_kwargs={"patch_size": 14, "img_size": 224, "num_classes": 0},
        ),
        "Phikon": ModelSpec(
            backend="hf_auto",
            model_id="owkin/phikon",
            extract="cls",
        ),
        "Phikon-v2": ModelSpec(
            backend="hf_auto",
            model_id="owkin/phikon-v2",
            extract="cls",
        ),
        "Hibou-L": ModelSpec(
            backend="hf_auto",
            model_id="histai/hibou-L",
            extract="cls",
        ),
        "Hibou-B": ModelSpec(
            backend="hf_auto",
            model_id="histai/hibou-b",
            extract="cls",
        ),
        "mSTAR": ModelSpec(
            backend="timm",
            model_id="hf-hub:Wangyh/mSTAR",
            extract="cls",
            timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
        ),
        "DINOv2-B": ModelSpec(
            backend="timm",
            model_id="vit_base_patch14_dinov2.lvd142m",
            extract="cls",
            timm_kwargs={"dynamic_img_size": True},
        ),
        "GPFM": ModelSpec(
            backend="gpfm",
            model_id="majiabo/GPFM",
            extract="cls",
        ),
        "MUSK": ModelSpec(
            backend="musk",
            model_id="hf_hub:xiangjx/musk",
            extract="ms_aug",
            mixed_precision=True,
        ),
        "GenBio-PathFM": ModelSpec(
            backend="genbio",
            model_id="genbio-ai/genbio-pathfm",
            extract="raw",
        ),
    }
