from flax import nnx
from functools import partial 
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, List, Dict, Any

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

class Linear(nnx.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1.0, init_bias=0.0):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init_mode = init_mode
        self.init_weight = init_weight
        self.init_bias = init_bias
        
        # Initialize weights
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        key = nnx.make_rng('params')
        w_key, b_key = jax.random.split(key)
        
        self.weight = weight_init([out_features, in_features], **init_kwargs, key=w_key) * init_weight
        if bias:
            self.bias_param = weight_init([out_features], **init_kwargs, key=b_key) * init_bias
        else:
            self.bias_param = None

    def __call__(self, x):
        x = jnp.dot(x, self.weight.T)
        if self.bias_param is not None:
            x = x + self.bias_param
        return x

class Conv2d(nnx.Module):
    """2D convolution layer with optional up/downsampling."""
    
    def __init__(self,rngs,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1.0, init_bias=0.0,
    ):
        assert not (up and down)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.bias = bias
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.resample_filter = resample_filter
        
        # Initialize filter for up/downsampling
        f = jnp.array(resample_filter, dtype=jnp.float32)
        f_outer = jnp.outer(f, f)
        self.f = f_outer.reshape(1, 1, *f_outer.shape) / jnp.sum(f)**2
        # self.f = jnp.outer(f, f).reshape(1, 1, *f.shape) / jnp.sum(f)**2
        
        # Initialize weights and bias
        init_kwargs = dict(mode=init_mode, 
                          fan_in=in_channels*kernel*kernel,
                          fan_out=out_channels*kernel*kernel)
        
        w_key, b_key = jax.random.split(rngs)
        
        if kernel:
            self.weight = weight_init([out_channels, in_channels, kernel, kernel], 
                                    **init_kwargs, key=w_key) * init_weight
            if bias:
                self.bias_param = weight_init([out_channels], **init_kwargs, key=b_key) * init_bias
            else:
                self.bias_param = None
        else:
            self.weight = None
            self.bias_param = None

    def __call__(self, x):
        w_pad = self.kernel // 2 if self.weight is not None else 0
        f_pad = (len(self.resample_filter) - 1) // 2 if self.f is not None else 0
        
        if self.fused_resample and self.up and self.weight is not None:
            x = jax.lax.conv_transpose(x, self.f * 4, 
                                     strides=(2, 2),
                                     padding='SAME',
                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            x = jax.lax.conv(x, self.weight, 
                           strides=(1, 1),
                           padding='SAME',
                           dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        elif self.fused_resample and self.down and self.weight is not None:
            x = jax.lax.conv(x, self.weight,
                           strides=(1, 1),
                           padding='SAME',
                           dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            x = jax.lax.conv(x, self.f,
                           strides=(2, 2),
                           padding='SAME',
                           dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        else:
            if self.up:
                x = jax.lax.conv_transpose(x, self.f * 4,
                                         strides=(2, 2),
                                         padding='SAME',
                                         dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            if self.down:
                x = jax.lax.conv(x, self.f,
                               strides=(2, 2),
                               padding='SAME',
                               dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            if self.weight is not None:
                x = jax.lax.conv(x, self.weight,
                               strides=(1, 1),
                               padding='SAME',
                               dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        
        if self.bias_param is not None:
            x = x + self.bias_param.reshape(1, 1, 1, -1)
        return x

class GroupNorm(nnx.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        self.num_channels = num_channels
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        
        # Initialize scale and bias
        self.scale = jnp.ones(num_channels)
        self.bias = jnp.zeros(num_channels)

    def __call__(self, x):
        # Reshape for group norm
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height, width, self.num_groups, channels // self.num_groups)
        
        # Compute mean and variance
        mean = jnp.mean(x, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x, axis=(1, 2, 4), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(batch, height, width, channels)
        
        # Apply scale and bias
        x = x * self.scale.reshape(1, 1, 1, -1) + self.bias.reshape(1, 1, 1, -1)
        return x

class PositionalEmbedding(nnx.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def __call__(self, x):
        freqs = jnp.arange(0, self.num_channels//2, dtype=jnp.float32)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        
        x = jnp.outer(x, freqs)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x

class FourierEmbedding(nnx.Module):
    def __init__(self, num_channels, scale=16.0):
        self.num_channels = num_channels
        self.scale = scale
        
        # Initialize frequencies
        key = nnx.make_rng('params')
        self.freqs = jax.random.normal(key, (num_channels // 2,)) * scale

    def __call__(self, x):
        x = jnp.outer(x, 2 * jnp.pi * self.freqs)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x 