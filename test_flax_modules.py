import os
import sys
import jax
import jax.numpy as jnp
import torch
import numpy as np
import pytest
import flax.nnx as nnx

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import modules
from models.modules import (
    Linear as TorchLinear,
    Conv2d as TorchConv2d,
    GroupNorm as TorchGroupNorm,
    PositionalEmbedding as TorchPositionalEmbedding,
    FourierEmbedding as TorchFourierEmbedding
)
from models.nnx_modules_new import (
    Linear,
    Conv2d,
    GroupNorm,
    PositionalEmbedding,
    FourierEmbedding
)

def compare_tensors(torch_tensor, jax_tensor, rtol=1e-5, atol=1e-5):
    """Compare PyTorch and JAX tensors with given tolerances"""
    torch_tensor = torch_tensor.detach().numpy()
    jax_tensor = np.array(jax_tensor)
    return np.allclose(torch_tensor, jax_tensor, rtol=rtol, atol=atol)

def test_linear():
    # Initialize modules
    torch_linear = TorchLinear(10, 20, init_mode='kaiming_normal')
    flax_linear = Linear(10, 20, init_mode='kaiming_normal')
    
    # Create test input
    x = torch.randn(1, 10)
    x_jax = jnp.array(x.numpy())
    
    # Forward pass
    torch_out = torch_linear(x)
    flax_out = flax_linear(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)

def test_conv2d():
    # Initialize modules
    torch_conv = TorchConv2d(3, 6, kernel=3, init_mode='kaiming_normal')
    flax_conv = Conv2d(3, 6, kernel=3, init_mode='kaiming_normal')
    
    # Create test input
    x = torch.randn(1, 3, 32, 32)
    x_jax = jnp.array(x.permute(0, 2, 3, 1).numpy())  # NCHW to NHWC
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out.permute(0, 2, 3, 1), flax_out)  # NCHW to NHWC

def test_groupnorm():
    # Initialize modules
    torch_gn = TorchGroupNorm(32, num_groups=8)
    flax_gn = GroupNorm(32, num_groups=8)
    
    # Create test input
    x = torch.randn(1, 32, 32, 32)
    x_jax = jnp.array(x.permute(0, 2, 3, 1).numpy())  # NCHW to NHWC
    
    # Forward pass
    torch_out = torch_gn(x.permute(0, 2, 3, 1))  # NHWC
    flax_out = flax_gn(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)

def test_positional_embedding():
    # Initialize modules
    torch_pe = TorchPositionalEmbedding(64)
    flax_pe = PositionalEmbedding(64)
    
    # Create test input
    x = torch.tensor([0.5])
    x_jax = jnp.array(x.numpy())
    
    # Forward pass
    torch_out = torch_pe(x)
    flax_out = flax_pe(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)

def test_fourier_embedding():
    # Initialize modules
    torch_fe = TorchFourierEmbedding(64)
    flax_fe = FourierEmbedding(64)
    
    # Create test input
    x = torch.tensor([0.5])
    x_jax = jnp.array(x.numpy())
    
    # Forward pass
    torch_out = torch_fe(x)
    flax_out = flax_fe(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)

if __name__ == '__main__':
    pytest.main([__file__]) 