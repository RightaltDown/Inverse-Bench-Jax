from dataclasses import field
from typing import List, Optional
import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
    conv: str | None = None
    attn: str | None = None
    time_embed: str | None = None
    norm: str | None = None
    resblocks: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        return tuple(getattr(self, key) for key in keys)


@dataclasses.dataclass(unsafe_hash=True)
class TrainingConfig:
    # TODO: Add other training configs
    dataset_name: str = "navier_stokes"
    exp_dir: str = "exps/pretrain"
    exp_name: str = "EDM-NS2d"
    num_train_workers: int = 0
    batch_size: int = 4  # 256 before but temprorarily reduced to 4 for testing
    num_steps: int = 100_000
    warmup_steps: int = 500
    lr_min: float = 0.00001
    lr_max: float = 0.0001
    resume: Optional[str] = None
    resume_path: Optional[str] = None
    resume_step: int = 0
    seed: int = 42

    ema_halflife_nimg: int = 204800
    ema_rampup_ratio: float = 0.05
    ema_decay: float = 0.9999
    restore_checkpoints: bool = False


@dataclasses.dataclass(unsafe_hash=True)
class ParallelismConfig:
    data_sharding: tuple[str, ...] = ("data",)
    mesh_axes: tuple[str, ...] = ("data", "fsdp", "tensor")
    axis_rules: MeshRules = MeshRules(
        # For UNet Components
        conv="tensor",  # Shard convolutional layers across tensor dimension
        attn="tensor",  # Shard attention blocks across tensor dimension
        time_embed="fsdp",  # Shard time embeddings across FSDP dimension
        norm="fsdp",  # Normalization layers using FSDP
        resblocks="fsdp",  # Residual blocks parameters using FSDP
    )
    dcn_data_parallelism: int = -1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1
    ici_data_parallelism: int = 1
    ici_fsdp_parallelism: int = -1
    ici_tensor_parallelism: int = 1

    # TODO: Uncomment these when we have a better understanding of the parallelism
    # Cross-node parallelism (slower communication)
    # dcn_data_parallelism = 2      # Split batches across nodes
    # dcn_fsdp_parallelism = 1      # No cross-node parameter sharding
    # dcn_tensor_parallelism = 1    # No cross-node tensor sharding

    # # Within-node parallelism (faster communication)
    # ici_data_parallelism = 1      # No additional data parallelism within nodes
    # ici_fsdp_parallelism = 2      # Split parameters 2 ways within each node
    # ici_tensor_parallelism = 2    # Split tensor operations 2 ways within each node


@dataclasses.dataclass(unsafe_hash=True)
class LogConfig:
    wandb: bool = False
    project: str = "EKS-DM-NS2d"
    group: str = "training"
    exp_name: str = "EDM-NS2d"
    wandb_freq: int = 100
    print_freq: int = 100
    sample_freq: int = 500
    log_every_steps: int = 100
    save_freq: int = 5000
    save_checkpoints: bool = True


@dataclasses.dataclass(unsafe_hash=True)
class ModelConfig:
    model_type: str = "DhariwalUNet"
    img_resolution: int = 128
    img_channels: int = 1
    label_dim: int = 0
    model_channels: int = 128
    channel_mult: List[int] = field(default_factory=lambda: [1, 1, 1, 2, 2])
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    num_blocks: int = 1
    dropout: float = 0.0


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    training: TrainingConfig = TrainingConfig()
    log: LogConfig = LogConfig()
    model: ModelConfig = ModelConfig()
    parallelism: ParallelismConfig = ParallelismConfig()
    # Add other sections as needed (loss, scheduler, data, etc.)


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
