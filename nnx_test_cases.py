from flax import nnx
from functools import partial 
import numpy as np
import jax
import jax.numpy as jnp

# Import modules
from models.modules import (
    Linear as TorchLinear,
    Conv2d as TorchConv2d,
    GroupNorm as TorchGroupNorm,
    PositionalEmbedding as TorchPositionalEmbedding,
    FourierEmbedding as TorchFourierEmbedding
)
from models.nnx_modules import (
    Linear,
    Conv2d,
    # GroupNorm,
    # PositionalEmbedding,
    # FourierEmbedding
)


import torch
TOL = 1e-4


def compare_tensors(torch_tensor, jax_tensor, rtol=1e-5, atol=1e-5, debug=False):
    """Compare PyTorch and JAX tensors with given tolerances"""
    torch_tensor = torch_tensor.detach().numpy()
    jax_tensor = np.array(jax_tensor)
    
    if not debug: return np.allclose(torch_tensor, jax_tensor, rtol=rtol, atol=atol)
    
    print(f"Shapes: JAX {jax_tensor.shape}, PyTorch {torch_tensor.shape}")
    assert torch_tensor.shape == jax_tensor.shape, "Shape mismatch"

    diff = np.abs(torch_tensor - jax_tensor)
    print(f"Mean: {np.mean(diff)}")
    if torch_tensor.size < 100:
        print("JAX output:\n", torch_tensor)
        print("PyTorch output:\n", jax_tensor)
        print("Difference:\n", diff)
    return np.allclose(torch_tensor, jax_tensor, rtol=rtol, atol=atol)


def test_linear_module():
    # Initialize the module
    rngs = nnx.Rngs(0, params=1) # this doesn't really matter when running the modules in test mode since we use numpy to compare differences
    torch_linear = TorchLinear(10, 20, init_mode='test')
    flax_linear = Linear(rngs, 10, 20, init_mode='test')
    
    x = torch.randn(1,10)
    x_jax = jnp.array(x.numpy())
    
    torch_out = torch_linear(x)
    flax_out = flax_linear(x_jax)
    
    assert compare_tensors(torch_out, flax_out)

def test_conv2d():
    # Create inputs and keys
    rngs = nnx.Rngs(0, params=1)
    
    # Create test input
    
    x = torch.randn(2, 4, 32, 32)
    x_jax = jnp.array(x.numpy())
    
    # default
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test')
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test')
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out), "default"
    
    # down
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test', down=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test', down=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "down"
    
    # up
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test', up=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test', up=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "down"
    
    # fused_resample + up
    torch_conv = TorchConv2d(4, 6, kernel=2, init_mode='test', fused_resample=True, up=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=2, init_mode='test', fused_resample=True, up=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "fused resample + up"
    
    # fused_resample + down
    torch_conv = TorchConv2d(4, 6, kernel=2, init_mode='test', fused_resample=True, down=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=2, init_mode='test', fused_resample=True, down=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "fused resample + down"
    
def temp():
    # Define parameters
    batch_size = 2
    in_channels = 3  # RGB image
    height, width = 8, 8  # Small test image

    # Create a simple test input: a batch with random values
    x = torch.randn(batch_size, in_channels, height, width)
    x_jax = jnp.array(x.numpy())
    print(f"Input shape: {x.shape}")

    # Create the filter as described
    resample_filter = [1, 1]
    f = torch.as_tensor(resample_filter, dtype=torch.float32)
    f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
    
    f_jax = jnp.array(resample_filter, dtype=jnp.float32)
    f_outer = jnp.outer(f_jax, f_jax)
    f_jax = f_outer.reshape(1, 1, *f_outer.shape) / jnp.sum(f_jax)**2
    assert compare_tensors(f, f_jax)

    # Tile the filter
    f_tiled = f.tile([in_channels, 1, 1, 1])
    f_jax_tiled = jnp.tile(f_jax , (in_channels, 1, 1, 1))
    print(f"Tiled filter shape: {f_tiled.shape}")
    assert compare_tensors(f_tiled, f_jax_tiled)

    # Calculate proper padding
    f_pad = (f.shape[-1] - 1) // 2
    f_pad_jax = (f_jax.shape[-1] - 1) // 2
    print(f"Padding: {f_pad}")

    # Apply convolution
    y = torch.nn.functional.conv2d(
        x, 
        f_tiled, 
        groups=in_channels, 
        stride=2, 
        padding=f_pad
    )
    
    y_jax = jax.lax.conv_general_dilated(
            x_jax, f_jax_tiled, 
            window_strides=(2, 2), 
            padding=[(f_pad_jax, f_pad_jax)] * 2,
            feature_group_count=in_channels
        )
    
    assert compare_tensors(y, y_jax)
    
    
    
    
    

if __name__ == "__main__":
    # test_linear_module()
    test_conv2d()
    # temp()