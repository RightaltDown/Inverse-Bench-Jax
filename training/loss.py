"""
This file implements the loss functions used in the training of the diffusion models.
"""

import torch
from utils import persistence
from piq import psnr, SSIMLoss
from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
from flax import nnx

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


# Type for loss factory functions
LossFactory = Callable[[Dict[str, Any]], Callable]

# Registry of loss factories
LOSS_REGISTRY: Dict[str, LossFactory] = {}


def register_loss(name: str):
    def decorator(factor_fn: LossFactory) -> LossFactory:
        LOSS_REGISTRY[name] = factor_fn
        return factor_fn

    return decorator


def get_loss(loss_name: str, config: Dict[str, Any]) -> Callable:
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {loss_name} not found in registry")
    return LOSS_REGISTRY[loss_name](**config)


# def edm_loss_fn(rngs, model, images, sigma_data=0.5, P_mean=-1.2, P_std=1.2, labels=None, augment_pipe=None):
#     rng_sigma, rng_noise = jax.random.split(rngs.loss())
#     rnd_normal = jax.random.normal(rng_sigma, (images.shape[0], 1, 1, 1))
#     sigma = jnp.exp(rnd_normal * P_std + P_mean)
#     weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

#     # Apply augmentation if provided
#     y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

#     # Generate noise to add to image
#     noise = jax.random.normal(rng_noise, y.shape)
#     noised_input = y + noise * jnp.reshape(sigma, (-1, 1, 1, 1))

#     # Get denoised prediction from model
#     D_yn = model(noised_input, sigma, labels, augment_labels=augment_labels)

#     # Calculate MSE loss with importance weighting
#     mse = ((D_yn - y) ** 2)
#     loss = weight * jnp.mean(mse, axis=(1, 2, 3))  # Average over spatial dimensions first
#     return jnp.mean(loss)  # Then average over batch


def edm_loss_fn(
    model,
    images,
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    rngs: nnx.Rngs = None,
    labels=None,
    augment_pipe=None,
):
    rng_sigma, rng_noise = jax.random.split(rngs.loss())
    rnd_normal = jax.random.normal(rng_sigma, (images.shape[0], 1, 1, 1))
    sigma = jnp.exp(rnd_normal * P_std + P_mean)
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2

    # Apply augmentation if provided
    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )

    # Generate noise to add to image
    noise = jax.random.normal(rng_noise, y.shape)
    noised_input = y + noise * sigma

    # Get denoised prediction from model
    D_yn = model(noised_input, sigma, labels, augment_labels=augment_labels, rngs=rngs)

    # Calculate MSE loss with importance weighting
    mse = (D_yn - y) ** 2
    loss = weight * jnp.mean(
        mse, axis=(1, 2, 3)
    )  # Average over spatial dimensions first
    return jnp.mean(loss)  # Then average over batch


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------


class DynamicRangePSNRLossForTraining:
    def _mse(self, yhats, ys):
        dims = tuple(range(1, len(ys.shape)))
        return ((yhats - ys) ** 2).mean(dim=dims)

    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:  # complex input: convert to magnitude image
            yhats = (
                torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous())
                .type(torch.complex128)
                .abs()
                .unsqueeze(1)
            )
        if ys.shape[1] == 2:
            ys = (
                torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous())
                .type(torch.complex128)
                .abs()
                .unsqueeze(1)
            )
        mse = self._mse(yhats, ys)
        data_range = [y.max() for y in ys]
        data_range = torch.stack(data_range, dim=0)
        psnr = 10 * torch.log10((data_range**2) / mse)
        return -torch.mean(psnr)


class DynamicRangePSNRLoss:
    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:  # complex input: convert to magnitude image
            yhats = (
                torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous())
                .type(torch.complex128)
                .abs()
                .unsqueeze(1)
            )
        if ys.shape[1] == 2:
            ys = (
                torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous())
                .type(torch.complex128)
                .abs()
                .unsqueeze(1)
            )
        return -torch.mean(
            torch.stack(
                [
                    psnr(
                        yhat.clip(0, y.max()).unsqueeze(0),
                        y.unsqueeze(0),
                        data_range=y.max(),
                    )
                    for yhat, y in zip(yhats, ys)
                ]
            )
        )


class DynamicRangeSSIMLoss:
    def __init__(self):
        self.ssim_loss = SSIMLoss()

    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:  # complex input: convert to magnitude image
            yhats = (
                torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous())
                .type(torch.complex128)
                .abs()
                .unsqueeze(1)
            )
        if ys.shape[1] == 2:
            ys = (
                torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous())
                .type(torch.complex128)
                .abs()
                .unsqueeze(1)
            )
        return torch.mean(
            torch.stack(
                [
                    SSIMLoss(data_range=y.max())(
                        yhat.clip(0, y.max()).unsqueeze(0), y.unsqueeze(0)
                    )
                    for yhat, y in zip(yhats, ys)
                ]
            )
        )


@persistence.persistent_class
class MRILoss:
    def __init__(self, forward_op, loss_type="psnr_loss", target_type="mvue_abs"):
        self.forward_op = forward_op
        if loss_type == "psnr_loss":
            self.loss = DynamicRangePSNRLossForTraining()
        elif loss_type == "ssim_loss":
            self.loss = SSIMLoss(spatial_dims=2)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        assert target_type in [
            "mvue_abs",
            "rss",
        ], f"Invalid target_type: {target_type}. Choose from mvue, rss"
        self.target_type = target_type

    def __call__(self, net, data):
        observation = self.forward_op(data)
        recon = net(observation, self.forward_op.mask.unsqueeze(-1))
        if self.target_type == "rss":
            target = data["rss"]
        else:
            target = data["mvue"].type(torch.complex128).abs()
        return self.loss(recon, target)
