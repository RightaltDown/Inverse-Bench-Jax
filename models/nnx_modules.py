from flax import nnx
from functools import partial 
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

def weight_init(key, shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * jax.random.normal(key, shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * jax.random.normal(key, shape)
    if mode == 'test': 
        return jnp.ones(shape)
        np.random.seed(10) 
        output = np.sqrt(1 / fan_in) * (jnp.asarray(np.random.randn(*shape))* 2 - 1)
        # reshape from: [out_channels, in_channels, kernel, kernel]
        # to: [kernel, kernel, in_channels, out_channels] - HWIO
        return jnp.transpose(output, (3, 2, 1, 0))
        # return np.sqrt(1 / fan_in) * (jnp.asarray(np.random.randn(*shape))* 2 - 1)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(nnx.Module):
    def __init__(self, rngs, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        
        self.rngs = rngs
        self.in_features = in_features
        self.out_features = out_features
        
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        weight_key = self.rngs
        self.weight = nnx.Param(weight_init(weight_key, [out_features, in_features], **init_kwargs) * init_weight)
        if bias: 
            bias_key = self.rngs
            self.bias = nnx.Param(weight_init(bias_key, [out_features], **init_kwargs) * init_bias)
        else:
            self.bias = None
    
    def __call__(self, x):
        x = jnp.matmul(x, self.weight.value.transpose())
        if self.bias is not None:
            x += jnp.reshape(self.bias.value, (1,) * (x.ndim - 1) + (-1,))
        return x

class Conv2d(nnx.Module):
    """2D convolution layer with optional up/downsampling."""
    
    def __init__(self, rngs,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0
    ):
        
        assert not (up and down), "Cannot upsample and downsample simultaneously"
        super().__init__()
        
        self.rngs = rngs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        
        # Initialize weights in HWIO format for NHWC
        init_kwargs = dict(
            mode=init_mode, 
            fan_in=in_channels*kernel*kernel, 
            fan_out=out_channels*kernel*kernel
        )
        weight_key = self.rngs.params()
        # self.weight = nnx.Param(weight_init(weight_key, [out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        # Create weight in HWIO format
        weight = weight_init(weight_key, [kernel, kernel, in_channels, out_channels], **init_kwargs) * init_weight
        self.weight = nnx.Param(weight) if kernel else None
        
        if kernel and bias:
            bias_key = self.rngs.params()
            self.bias = nnx.Param(weight_init(bias_key, [out_channels], **init_kwargs) * init_bias)
        else: 
            self.bias = None
        
        # Create resample filter in HWIO format
        f = jnp.array(resample_filter, dtype=jnp.float32)
        f_outer = jnp.outer(f, f)
        f_reshaped = f_outer.reshape(f_outer.shape[0], f_outer.shape[1], 1, 1)
        self.resample_filter = nnx.Param(
                f_reshaped / (jnp.sum(f) ** 2), 
                trainable=False
        )
        # f = f_outer.reshape(1, 1, *f_outer.shape) / jnp.sum(f)**2
        # self.resample_filter = nnx.Param(f, trainable=False)
        
    
    def __call__(self, x):
        # Get parameters
        w = self.weight.value.astype(x.dtype) if self.weight is not None else None
        b = self.bias.value.astype(x.dtype) if self.bias is not None else None
        f = self.resample_filter.value.astype(x.dtype) if self.resample_filter is not None else None
        
        # Calculate padding
        # w_pad = w.shape[-1] // 2 if w is not None else 0
        # f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        w_pad = w.shape[0] // 2 if w is not None else 0  # Using first dimension for HWIO format
        f_pad = (f.shape[0] - 1) // 2 if f is not None else 0
        
        # Handle different convolution cases
        if self.fused_resample and self.up and w is not None:
            # Fused upsampling + convolution
            # print("Fused upsampling + convolution")
            # f_up = jnp.tile(f * 4, (self.in_channels, 1, 1, 1))
            f_up = jnp.tile(f * 4, (1, 1, 1, self.in_channels))  # HWIO format
            x = jax.lax.conv_general_dilated(
                x, 
                f_up,
                window_strides=(1, 1),
                padding=[(max(f_pad - w_pad + 1, 1), max(f_pad - w_pad + 1, 1))] * 2,
                lhs_dilation=(2, 2),
                rhs_dilation=(1, 1),
                feature_group_count=self.in_channels,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
            x = jax.lax.conv_general_dilated(
                x, w, 
                window_strides=(1, 1), 
                padding=[(max(w_pad - f_pad, 0), max(w_pad - f_pad, 0))] * 2,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
            
        elif self.fused_resample and self.down and w is not None:
            # Fused convolution + downsampling
            # print("Fused convolution + downsampling")
            x = jax.lax.conv_general_dilated(
                x, w, 
                window_strides=(1, 1), 
                padding=[(w_pad + f_pad, w_pad + f_pad)] * 2,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
            # f_down = jnp.tile(f, (self.out_channels, 1, 1, 1))
            f_down = jnp.tile(f, (1, 1, 1, self.out_channels))  # [H, W, out_channels, 1]
            x = jax.lax.conv_general_dilated(
                x, f_down, 
                window_strides=(2, 2), 
                padding=[(0, 0)] * 2,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=self.out_channels
            )
            
        else:
            # Handle separate operations
            # print("Handle separate operations")
            if self.up:
                # print("self.up")
                # f_up = jnp.tile(f * 4, (self.in_channels, 1, 1, 1))
                f_up = jnp.tile(f * 4, (1, 1, 1,self.in_channels))  # HWIO format
                x = jax.lax.conv_general_dilated(
                    x, 
                    f_up,
                    window_strides=(1, 1),
                    padding=[(f_pad + 1, f_pad + 1)] * 2,
                    lhs_dilation=(2, 2),
                    rhs_dilation=(1, 1),
                    feature_group_count=self.in_channels,
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC')
                )
                
            if self.down:
                # print("self.down")
                # f_down = jnp.tile(f, (self.in_channels, 1, 1, 1))
                f_down = jnp.tile(f, (1, 1, 1, self.in_channels))  # HWIO format
                x = jax.lax.conv_general_dilated(
                    x, f_down, 
                    window_strides=(2, 2), 
                    padding=[(f_pad, f_pad)] * 2,
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                    feature_group_count=self.in_channels
                )
                
            if w is not None:
                # print("w")
                x = jax.lax.conv_general_dilated(
                    x, w, 
                    window_strides=(1, 1), 
                    padding=[(w_pad, w_pad)] * 2,
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC')
                )
        
        # Add bias if needed
        if b is not None:
            # x = jnp.add(x, b.reshape(1, -1, 1, 1))
            x = jnp.add(x, b.reshape(1, 1, 1, -1))  # Reshape bias for NHWC format
        return x

class GroupNorm(nnx.Module):
    def __init__(self, rngs, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        
        # Create the internal GroupNorm module
        self.norm = nnx.GroupNorm(
            num_features=num_channels,
            num_groups=self.num_groups,
            epsilon=eps,
            use_bias=True,
            use_scale=True,
            rngs=rngs
        )
        
    def __call__(self, x):
        return self.norm(x)
        
class UNetBlock(nnx.Module):
    def __init__(self, rngs,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(rngs=rngs, num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(rngs=rngs, in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(rngs=rngs, in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(rngs=rngs, num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(rngs=rngs, in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(rngs, in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(rngs, num_channels=out_channels, eps=eps)
            attn_init = init_attn if init_attn is not None else init
            self.qkv = Conv2d(rngs, in_channels=out_channels, out_channels=out_channels*3, kernel=1, **attn_init)
            self.proj = Conv2d(rngs, in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def __call__(self, x, emb, train=True):
        # Input is already in NHWC format
        orig = x
        
        # First conv block
        x = self.norm0(x)
        x = jax.nn.silu(x)
        x = self.conv0(x)

        # Conditioning
        params = self.affine(emb)
        # params = params.reshape(*params.shape[:-1], -1, 1, 1)  # Add spatial dimensions
        params = params.reshape(params.shape[0], 1, 1, -1)
        
        # Apply adaptive scaling or shift
        x = self.norm1(x)
        if self.adaptive_scale:
            scale, shift = jnp.split(params, 2, axis=-1)
            x = shift + x * (scale + 1)
        else:
            x = x + params
        x = jax.nn.silu(x)

        # Second conv block
        if self.dropout > 0 and train:
            key = self.make_rng('dropout')
            x = jax.random.bernoulli(key, 1 - self.dropout, x.shape) * x / (1 - self.dropout)
        x = self.conv1(x)
        
        # Skip connection
        if self.skip is not None:
            orig = self.skip(orig)
        x = x + orig
        x = x * self.skip_scale

        # Self attention block
        if self.num_heads:
            identity = x
            x = self.norm2(x)
            qkv = self.qkv(x)
            
            # Reshape for attention
            B, H, W, C = qkv.shape
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1)
            qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, H*W, C//num_heads)
            q, k, v = qkv
            
            # Scaled dot product attention
            scale = (q.shape[-1] ** -0.5)
            attention = jax.nn.softmax((q @ jnp.transpose(k, (0, 1, 3, 2))) * scale, axis=-1)
            x = (attention @ v).transpose(0, 1, 3, 2)  # (B, num_heads, H*W, C//num_heads)
            
            # Reshape back
            x = x.reshape(B, H, W, -1)
            x = self.proj(x)
            x = x + identity
            x = x * self.skip_scale
        
        return x

class PositionalEmbedding(nnx.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
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
    def __init__(self, rngs, num_channels, scale=16):
        super().__init__()
        key = rngs.params()
        self.freqs = nnx.Param(jax.random.normal(key, (num_channels // 2,)) * scale)

    def __call__(self, x):
        x = jnp.outer(x, 2 * jnp.pi * self.freqs.value)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x
         
