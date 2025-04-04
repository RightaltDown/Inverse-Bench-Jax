import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out, key):
    if mode == 'xavier_uniform':
        scale = jnp.sqrt(6 / (fan_in + fan_out))
        return scale * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == 'xavier_normal':
        scale = jnp.sqrt(2 / (fan_in + fan_out))
        return scale * jax.random.normal(key, shape)
    if mode == 'kaiming_uniform':
        scale = jnp.sqrt(3 / fan_in)
        return scale * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == 'kaiming_normal':
        scale = jnp.sqrt(1 / fan_in)
        return scale * jax.random.normal(key, shape)
    if mode == 'test':
        np.random.seed(10)
        scale = jnp.sqrt(1 / fan_in)
        return scale * (jnp.array(np.random.randn(*shape)) * 2 - 1)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True
    init_mode: str = 'kaiming_normal'
    init_weight: float = 1.0
    init_bias: float = 0.0

    @nn.compact
    def __call__(self, x):
        init_kwargs = dict(mode=self.init_mode, fan_in=self.in_features, fan_out=self.out_features)
        
        # Initialize weights
        key = self.make_rng('params')
        w_key, b_key = jax.random.split(key)
        w = weight_init([self.out_features, self.in_features], **init_kwargs, key=w_key) * self.init_weight
        
        # Initialize bias if needed
        b = None
        if self.bias:
            b = weight_init([self.out_features], **init_kwargs, key=b_key) * self.init_bias
        
        # Apply linear transformation
        x = jnp.dot(x, w.T)
        if b is not None:
            x = x + b
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel: int
    bias: bool = True
    up: bool = False
    down: bool = False
    resample_filter: List[int] = [1, 1]
    fused_resample: bool = False
    init_mode: str = 'kaiming_normal'
    init_weight: float = 1.0
    init_bias: float = 0.0

    @nn.compact
    def __call__(self, x):
        assert not (self.up and self.down)
        
        # Initialize filter for up/downsampling
        f = jnp.array(self.resample_filter, dtype=jnp.float32)
        f = jnp.outer(f, f).reshape(1, 1, *f.shape) / jnp.sum(f)**2
        
        # Initialize weights and bias
        init_kwargs = dict(mode=self.init_mode, 
                          fan_in=self.in_channels*self.kernel*self.kernel,
                          fan_out=self.out_channels*self.kernel*self.kernel)
        
        key = self.make_rng('params')
        w_key, b_key = jax.random.split(key)
        
        w = None
        b = None
        if self.kernel:
            w = weight_init([self.out_channels, self.in_channels, self.kernel, self.kernel], 
                          **init_kwargs, key=w_key) * self.init_weight
            if self.bias:
                b = weight_init([self.out_channels], **init_kwargs, key=b_key) * self.init_bias
        
        # Apply convolutions
        w_pad = self.kernel // 2 if w is not None else 0
        f_pad = (len(self.resample_filter) - 1) // 2 if f is not None else 0
        
        if self.fused_resample and self.up and w is not None:
            x = jax.lax.conv_transpose(x, f * 4, 
                                     strides=(2, 2),
                                     padding='SAME',
                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            x = jax.lax.conv(x, w, 
                           strides=(1, 1),
                           padding='SAME',
                           dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        elif self.fused_resample and self.down and w is not None:
            x = jax.lax.conv(x, w,
                           strides=(1, 1),
                           padding='SAME',
                           dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            x = jax.lax.conv(x, f,
                           strides=(2, 2),
                           padding='SAME',
                           dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        else:
            if self.up:
                x = jax.lax.conv_transpose(x, f * 4,
                                         strides=(2, 2),
                                         padding='SAME',
                                         dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            if self.down:
                x = jax.lax.conv(x, f,
                               strides=(2, 2),
                               padding='SAME',
                               dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            if w is not None:
                x = jax.lax.conv(x, w,
                               strides=(1, 1),
                               padding='SAME',
                               dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        
        if b is not None:
            x = x + b.reshape(1, 1, 1, -1)
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(nn.Module):
    num_channels: int
    num_groups: int = 32
    min_channels_per_group: int = 4
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        num_groups = min(self.num_groups, self.num_channels // self.min_channels_per_group)
        
        # Initialize scale and bias
        scale = self.param('scale', nn.initializers.ones, (self.num_channels,))
        bias = self.param('bias', nn.initializers.zeros, (self.num_channels,))
        
        # Reshape for group norm
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height, width, num_groups, channels // num_groups)
        
        # Compute mean and variance
        mean = jnp.mean(x, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x, axis=(1, 2, 4), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(batch, height, width, channels)
        
        # Apply scale and bias
        x = x * scale.reshape(1, 1, 1, -1) + bias.reshape(1, 1, 1, -1)
        return x

#----------------------------------------------------------------------------
# Positional embedding

class PositionalEmbedding(nn.Module):
    num_channels: int
    max_positions: int = 10000
    endpoint: bool = False

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(0, self.num_channels//2, dtype=jnp.float32)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        
        x = jnp.outer(x, freqs)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x

#----------------------------------------------------------------------------
# Fourier embedding

class FourierEmbedding(nn.Module):
    num_channels: int
    scale: float = 16.0

    @nn.compact
    def __call__(self, x):
        freqs = self.param('freqs', 
                          lambda key: jax.random.normal(key, (self.num_channels // 2,)) * self.scale,
                          (self.num_channels // 2,))
        
        x = jnp.outer(x, 2 * jnp.pi * freqs)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x 