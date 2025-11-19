from __future__ import annotations
from flax import nnx
import jax.numpy as jnp
from models.nnx_unets import DhariwalUNet


import dataclasses
from typing import Any


Shape = tuple[int, ...]
Dtype = Any


_model_dict = {"DhariwalUNet": DhariwalUNet}


@dataclasses.dataclass(unsafe_hash=True)
class EDMPrecondConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    model_type: str
    img_resolution: int
    img_channels: int
    label_dim: int
    use_fp16: bool = False
    sigma_min: float = 0
    sigma_max: float = float("inf")
    sigma_data: float = 0.5
    model_channels: int = 128
    channel_mult: list[int] = dataclasses.field(default_factory=lambda: [1, 1, 1, 2, 2])
    attn_resolutions: list[int] = dataclasses.field(default_factory=lambda: [16])
    num_blocks: int = 1
    dropout: float = 0.0
    

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class EDMPrecond(nnx.Module):
    def __init__(
        self,
        img_resolution: int,  # Image resolution.
        img_channels: int,  # Number of color channels.
        label_dim: int = 0,  # Number of class labels, 0 = unconditional.
        use_fp16: bool = False,  # Execute the underlying model at FP16 precision?
        sigma_min: float = 0,  # Minimum supported noise level.
        sigma_max: float = float("inf"),  # Maximum supported noise level.
        sigma_data: float = 0.5,  # Expected standard deviation of the training data.
        model_type: str = "DhariwalUNet",  # Class name of the underlying model.
        rngs: nnx.Rngs = None,  # rngs for the model
        **model_kwargs: Any,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        

        self.model = _model_dict[model_type](
            rngs=rngs,
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    # train is usually False?
    def __call__(self, x, sigma, class_labels=None, train=True, **model_kwargs):
        x = jnp.asarray(x, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        dtype = jnp.float16 if self.use_fp16 else jnp.float32
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = jnp.zeros((1, self.label_dim), dtype=jnp.float32)
        else:
            class_labels = jnp.asarray(class_labels, dtype=jnp.float32).reshape(
                -1, self.label_dim
            )

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = jnp.sqrt(sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2))
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        F_x = self.model(jnp.asarray(c_in * x, dtype=dtype), c_noise.flatten(), class_labels=class_labels, train=train, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)
        return D_x

    def round_sigma(self, sigma):
        return jnp.asarray(sigma, dtype=jnp.float32)


_precond_dict = {"edm": EDMPrecond}


def get_model(name, **kwargs):
    return _precond_dict[name.lower()](**kwargs)
