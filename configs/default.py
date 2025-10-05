"""Default Hyperparameter configuration."""

import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
    # Simplified mesh rules for diffusion models
    # No need for complex transformer-specific rules
    data: str = "data"
    model: str = "model"


@dataclasses.dataclass(unsafe_hash=True)
class ParallelismConfig:
    mesh_axes_data: str = "data"

    dcn_data_parallelism: int = 1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1
    ici_data_parallelism: int = 4
    ici_fsdp_parallelism: int = 1
    ici_tensor_parallelism: int = 1

    data_sharding: tuple[str, ...] = ("data",)

    mesh_axes: tuple[str, ...] = ("data", "fsdp", "tensor")
    axis_rules: MeshRules = MeshRules(
        data="data",
        model="model",
    )


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    parallelism: ParallelismConfig = ParallelismConfig()

    # Model parameters
    model_type: str = "DhariwalUNet"
    img_resolution: int = 128
    img_channels: int = 1
    label_dim: int = 0
    model_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 1, 1, 2, 2)
    attn_resolutions: tuple[int, ...] = 16
    num_blocks: int = 1
    dropout: float = 0.0

    # Basic training parameters
    seed: int = 42
    num_train_steps: int = 100_000
    eval_every_steps: int = 1000
    per_device_batch_size: int = 4

    warmup_steps: int = 1000
    base_lr: float = 1e-4


# @dataclasses.dataclass(unsafe_hash=True)
# class ModelConfig:
#     model_type: str = "DhariwalUNet"
#     img_resolution: int = 128
#     img_channels: int = 1
#     label_dim: int = 0
#     model_channels: int = 128
#     channel_mult: List[int] = dataclasses.field(default_factory=lambda: [1, 1, 1, 2, 2])
#     attn_resolutions: List[int] = dataclasses.field(default_factory=lambda: [16])
#     num_blocks: int = 1
#     dropout: float = 0.0
#     use_fp16: bool = False
#     sigma_min: float = 0.0
#     sigma_max: float = float("inf")
#     sigma_data: float = 0.5


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
