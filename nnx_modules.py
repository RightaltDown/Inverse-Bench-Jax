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
        np.random.seed(10)
        return np.sqrt(1 / fan_in) * (jnp.asarray(np.random.randn(*shape)) * 2 - 1)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(nnx.Module):
    def __init__(self, rngs, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        
        self.rngs = rngs
        self.in_features = in_features
        self.out_features = out_features
        
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        weight_key = self.rngs.params()
        self.weight = nnx.Param(weight_init(weight_key, [out_features, in_features], **init_kwargs) * init_weight)
        if bias: 
            bias_key = self.rngs.params()
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
        
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        weight_key = self.rngs.params()
        self.weight = nnx.Param(weight_init(weight_key, [out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        if kernel and bias:
            bias_key = self.rngs.params()
            self.bias = nnx.Param(weight_init(bias_key, [out_channels], **init_kwargs) * init_bias)
        else: 
            self.bias = None
        
        f = jnp.array(resample_filter, dtype=jnp.float32)
        f_outer = jnp.outer(f, f)
        f = f_outer.reshape(1, 1, *f_outer.shape) / jnp.sum(f)**2
        self.resample_filter = nnx.Param(f, trainable=False)
        
    
    def __call__(self, x):
        # Get parameters
        w = self.weight.value.astype(x.dtype) if self.weight is not None else None
        b = self.bias.value.astype(x.dtype) if self.bias is not None else None
        f = self.resample_filter.value.astype(x.dtype) if self.resample_filter is not None else None
        
        # Calculate padding
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        
        # Handle different convolution cases
        if self.fused_resample and self.up and w is not None:
            # Fused upsampling + convolution
            print("Fused upsampling + convolution")
            f_up = jnp.tile(f * 4, (self.in_channels, 1, 1, 1))
            x = jax.lax.conv_general_dilated(
                    x, 
                    f_up,
                    window_strides=(1, 1),
                    padding=[(max(f_pad - w_pad + 1, 1), max(f_pad - w_pad + 1, 1))] * 2,
                    lhs_dilation=(2, 2),
                    rhs_dilation=(1, 1),
                    feature_group_count=self.in_channels,
                    dimension_numbers=('NCHW', 'OIHW', 'NCHW')
                )
            x = jax.lax.conv_general_dilated(
                x, w, 
                window_strides=(1, 1), 
                padding=[(max(w_pad - f_pad, 0), max(w_pad - f_pad, 0))] * 2,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            )
            
        elif self.fused_resample and self.down and w is not None:
            # Fused convolution + downsampling
            print("Fused convolution + downsampling")
            x = jax.lax.conv_general_dilated(
                x, w, 
                window_strides=(1, 1), 
                padding=[(w_pad + f_pad, w_pad + f_pad)] * 2,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            )
            f_down = jnp.tile(f, (self.out_channels, 1, 1, 1))
            x = jax.lax.conv_general_dilated(
                x, f_down, 
                window_strides=(2, 2), 
                padding=[(0, 0)] * 2,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                feature_group_count=self.out_channels
            )
            
        else:
            # Handle separate operations
            print("Handle separate operations")
            if self.up:
                print("self.up")
                f_up = jnp.tile(f * 4, (self.in_channels, 1, 1, 1))
                x = jax.lax.conv_general_dilated(
                    x, 
                    f_up,
                    window_strides=(1, 1),
                    padding=[(f_pad + 1, f_pad + 1)] * 2,
                    lhs_dilation=(2, 2),
                    rhs_dilation=(1, 1),
                    feature_group_count=self.in_channels,
                    dimension_numbers=('NCHW', 'OIHW', 'NCHW')
                )
                
                # f_up_expanded = jnp.tile(f_up, (1, 4, 1, 1))
                # x = jax.lax.conv_transpose(
                #     x, f_up_expanded, 
                #     strides=(2, 2), 
                #     padding=[(f_pad, f_pad)] * 2,
                #     dimension_numbers=('NCHW', 'OIHW', 'NCHW')
                # )
                
            if self.down:
                print("self.down")
                f_down = jnp.tile(f, (self.in_channels, 1, 1, 1))
                x = jax.lax.conv_general_dilated(
                    x, f_down, 
                    window_strides=(2, 2), 
                    padding=[(f_pad, f_pad)] * 2,
                    dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                    feature_group_count = self.in_channels
                )
                
            if w is not None:
                print("w")
                x = jax.lax.conv_general_dilated(
                    x, w, 
                    window_strides=(1, 1), 
                    padding=[(w_pad, w_pad)] * 2,
                    dimension_numbers=('NCHW', 'OIHW', 'NCHW')
                )
        
        # Add bias if needed
        if b is not None:
            x = jnp.add(x, b.reshape(1, -1, 1, 1))
        return x
