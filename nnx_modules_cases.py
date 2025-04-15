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
    FourierEmbedding as TorchFourierEmbedding,
    UNetBlock as TorchUNetBlock,
    weight_init as m_weight_init
)

from models.nnx_modules import (
    Linear,
    Conv2d,
    GroupNorm,
    PositionalEmbedding,
    FourierEmbedding,
    UNetBlock,
    weight_init as nnx_weight_init
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
    print(f"Mean difference: {np.mean(diff)}")
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
    print("Linear module test passed!")
    

def test_conv2d():
    # Create inputs and keys
    rngs = nnx.Rngs(0, params=1)
    
    # Create test input
    
    x = torch.randn(2, 4, 32, 32)
    x_jax = jnp.array(x.numpy())
    x_jax = jnp.transpose(x_jax, (0, 2, 3, 1))
    
    # default
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test')
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test')
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out), "default"
    print("default passed!")
    
    # down
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test', down=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test', down=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "down"
    print("down convolution passed!")
    
    # up
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test', up=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test', up=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "up"
    print("up convolution passed!")
    
    # fused_resample + up
    torch_conv = TorchConv2d(4, 6, kernel=2, init_mode='test', fused_resample=True, up=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=2, init_mode='test', fused_resample=True, up=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "fused resample + up"
    print("fused resample + up passed!")
    
    # fused_resample + down
    torch_conv = TorchConv2d(4, 6, kernel=2, init_mode='test', fused_resample=True, down=True)
    flax_conv = Conv2d(rngs, 4, 6, kernel=2, init_mode='test', fused_resample=True, down=True)
    
    # Forward pass
    torch_out = torch_conv(x)
    flax_out = flax_conv(x_jax)
    flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out) , "fused resample + down"
    print("fused resample + down passed!")
    print("Conv2D module test passed!")

def debug_conv2d_implementations():
    """Debug function to compare PyTorch and JAX Conv2D implementations."""
   
    # Create inputs and keys with fixed seeds
    rngs = nnx.Rngs(0, params=1)
    torch.manual_seed(1)  # Use fixed seed for PyTorch
    kernel = 3
    in_channels = 4 
    out_channels = 6
    init_kwargs = dict(
            mode='test', 
            fan_in=in_channels*kernel*kernel, 
            fan_out=out_channels*kernel*kernel
    )
    
    pytorch_weight_tensor = m_weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * 1
    # jax_weight_tensor = nnx_weight_init(rngs.params(), [out_channels, in_channels, kernel, kernel], **init_kwargs) * 1
    jax_weight_tensor = nnx_weight_init(rngs.params(), [kernel, kernel, in_channels, out_channels], **init_kwargs) * 1
    jax_weight_tensor = jnp.transpose(jax_weight_tensor, (3,2,1,0))
    
    # jax_weight_tensor = jnp.transpose(jax_weight_tensor, (3, 2, 1, 0))
    
    print(f"Shapes: {pytorch_weight_tensor.shape}, {jax_weight_tensor.shape}")
    weight_matching = compare_tensors(pytorch_weight_tensor, jax_weight_tensor)
    print(f"Weight quick test: {weight_matching}")
    
    
    # Create test input
    x = torch.randn(2, 4, 32, 32)
    x_jax = jnp.array(x.numpy())
    x_jax = jnp.transpose(x_jax, (0, 2, 3, 1))  # NCHW -> NHWC
    
    # Create both implementations with identical parameters
    torch_conv = TorchConv2d(4, 6, kernel=3, init_mode='test')
    flax_conv = Conv2d(rngs, 4, 6, kernel=3, init_mode='test')
    
    # Compare weights directly
    # Convert JAX weights to PyTorch format for comparison
    jax_weight = flax_conv.weight.value  # Shape: [H, W, I, O]
    jax_weight_pytorch_format = jnp.transpose(jax_weight, (3, 2, 1, 0))  # [O, I, H, W]
    
    print("Weight shapes:")
    print(f"PyTorch: {torch_conv.weight.shape}")
    print(f"JAX (original): {jax_weight.shape}")
    print(f"JAX (converted to PyTorch format): {jax_weight_pytorch_format.shape}")
    
    weight_match = compare_tensors(torch_conv.weight, jax_weight_pytorch_format, debug=True)
    print(f"Weights match: {weight_match}")
    print(f"Pytorch weights: {torch_conv.weight[0, 0, 0, 0:3].detach().numpy()}")
    print(f"Jax weights: {jax_weight[0, 0, 0, 0:3]}")
    
    # Compare bias
    if torch_conv.bias is not None and flax_conv.bias is not None:
        bias_match = compare_tensors(torch_conv.bias, flax_conv.bias.value)
        print(f"Biases match: {bias_match}")
    
    # Forward pass with simple input for easier debugging
    simple_x = torch.ones(2, 4, 8, 8)  # Simpler input for debugging
    simple_x_jax = jnp.ones((2, 8, 8, 4))  # NHWC format
    
    # Get outputs
    torch_out = torch_conv(simple_x)
    flax_out = flax_conv(simple_x_jax)
    flax_out_nchw = jnp.transpose(flax_out, (0, 3, 1, 2))  # NHWC -> NCHW
    
    print("Output shapes:")
    print(f"PyTorch: {torch_out.shape}")
    print(f"JAX (NHWC): {flax_out.shape}")
    print(f"JAX (converted to NCHW): {flax_out_nchw.shape}")
    
    # Compare first few values to see where differences occur
    print("\nFirst few output values:")
    print("PyTorch:", torch_out[0, 0, 0, 0:3].detach().numpy())
    print("JAX:", flax_out_nchw[0, 0, 0, 0:3])
    
    # Calculate relative error
    abs_diff = jnp.abs(torch_out.detach().numpy() - flax_out_nchw)
    max_diff = jnp.max(abs_diff)
    avg_diff = jnp.mean(abs_diff)
    
    print(f"\nMax absolute difference: {max_diff}")
    print(f"Average absolute difference: {avg_diff}")
    
    # Check convolution implementation details
    print("\nConvolution implementation details:")
    if hasattr(torch_conv, 'padding'):
        print(f"PyTorch explicit padding: {torch_conv.padding}")
    else:
        print("PyTorch using implicit padding")
    
    # Return comparison results
    return {
        "weights_match": weight_match,
        "outputs_match": compare_tensors(torch_out, flax_out_nchw),
        "max_diff": float(max_diff),
        "avg_diff": float(avg_diff)
    }


def test_group_norm():
    # Initialize parameters
    batch_size = 2
    num_channels = 32
    height = 16
    width = 16
    rngs = nnx.Rngs(0, params=1)
    
    # Create test input
    x = torch.randn(batch_size, num_channels, height, width)
    x_jax = jnp.array(x.numpy())
    # Convert from NCHW to NHWC for JAX
    x_jax = jnp.transpose(x_jax, (0, 2, 3, 1))
    
    # Create modules
    torch_gn = TorchGroupNorm(num_channels=num_channels)
    flax_gn = GroupNorm(num_channels=num_channels, rngs=rngs)
    
    # Forward pass
    torch_out = torch_gn(x)
    flax_out = flax_gn(x_jax)
    # Convert back to NCHW for comparison
    flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)
    print("GroupNorm module test passed!")

def nchw_to_nhwc(x):
    """Convert from NCHW format to NHWC format."""
    return jnp.transpose(x, (0, 2, 3, 1))

def nhwc_to_nchw(x):
    """Convert from NHWC format to NCHW format."""
    return jnp.transpose(x, (0, 3, 1, 2))

def temp():
    # Define parameters
    batch_size = 2
    in_channels = 3  # RGB image
    height, width = 8, 8  # Small test image

    # Create a simple test input: a batch with random values
    x = torch.randn(batch_size, in_channels, height, width)
    x_jax = jnp.array(x.numpy())
    x_jax = nchw_to_nhwc(x_jax)  # Convert to NHWC
    print(f"Input shape PyTorch (NCHW): {x.shape}")
    print(f"Input shape JAX (NHWC): {x_jax.shape}")

    # Create the filter as described
    resample_filter = [1, 1]
    
    # PyTorch filter (OIHW format): [out_channels, in_channels, height, width]
    f = torch.as_tensor(resample_filter, dtype=torch.float32)
    f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
    
    # JAX filter (HWIO format): [height, width, in_channels, out_channels]
    f_jax = jnp.array(resample_filter, dtype=jnp.float32)
    f_outer = jnp.outer(f_jax, f_jax)
    f_jax = f_outer / (jnp.sum(f_jax) ** 2)
    f_jax = f_jax.reshape(f_jax.shape[0], f_jax.shape[1], 1, 1)
    
    print(f"Filter PyTorch shape (OIHW): {f.shape}")
    print(f"Filter JAX shape (HWIO): {f_jax.shape}")

    # Tile the filter for grouped convolution
    # PyTorch: For grouped conv with groups=in_channels, we need [in_channels, 1, H, W]
    # The 1 here means "1 filter per group"
    f_tiled = f.repeat(in_channels, 1, 1, 1)  # Using repeat instead of tile for clarity
    
    # JAX: For grouped conv, we need [H, W, in_channels/groups, out_channels*groups]
    # For feature_group_count=in_channels, we need [H, W, 1, in_channels]
    f_jax_tiled = jnp.repeat(f_jax, in_channels, axis=3)  # Repeat in output channel dimension

    print(f"Tiled filter PyTorch shape: {f_tiled.shape}")
    print(f"Tiled JAX filter shape: {f_jax_tiled.shape}")

    # Calculate proper padding
    f_pad = (f.shape[-1] - 1) // 2
    print(f"Padding: {f_pad}")

    # Apply convolution
    # PyTorch: For groups=in_channels, each input channel uses its own filter
    y = torch.nn.functional.conv2d(
        x, 
        f_tiled, 
        groups=in_channels, 
        stride=2, 
        padding=f_pad
    )
    
    # JAX: For feature_group_count=in_channels, each input channel uses its own filter
    y_jax = jax.lax.conv_general_dilated(
        x_jax,
        f_jax_tiled, 
        window_strides=(2, 2), 
        padding=[(f_pad, f_pad)] * 2,
        feature_group_count=in_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    
    # Convert back to PyTorch format for comparison
    y_jax_nchw = nhwc_to_nchw(y_jax)
    
    print(f"Output PyTorch shape: {y.shape}")
    print(f"Output JAX shape (after conversion): {y_jax_nchw.shape}")
    
    # Compare results
    are_equal = compare_tensors(y, y_jax_nchw)
    print(f"Outputs match: {are_equal}")
    
    return are_equal

def conv_transpose_test1():
    # Define parameters
    batch_size = 2
    in_channels = 3  # RGB image
    height, width = 8, 8  # Small test image

    # Create a simple test input: a batch with random values
    x = torch.randn(batch_size, in_channels, height, width)
    # print(f"Input shape: {x.shape}")
    
    # Create resample filter
    resample_filter = [1,1]
    f = torch.as_tensor(resample_filter, dtype=torch.float32)
    f = f.ger(f).unsqueeze(0).unsqueeze(1)/ f.sum().square()
    f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
    
    # PyTorch implementation
    y = torch.nn.functional.conv_transpose2d(
        x, 
        f.mul(4).tile([in_channels, 1, 1, 1]), 
        groups=in_channels, 
        stride=2, 
        padding=f_pad
    )
    
    # JAX implementation
    x_jax = jnp.array(x.numpy())
    # Channel last for jax convention
    x_jax = jnp.transpose(x_jax, (0, 2, 3, 1))
    
    # Create filter in JAX
    f_jax = jnp.array(resample_filter, dtype=jnp.float32)
    f_jax = jnp.outer(f_jax, f_jax).reshape(1, 1, len(resample_filter), len(resample_filter)) / jnp.square(jnp.sum(f_jax))
    f_jax = f_jax * 4  # Multiply by 4 as in PyTorch

    # For grouped convolution, we need to process each channel separately
    y_jax_list = []
    for i in range(in_channels):
        # Select single channel
        x_channel = x_jax[..., i:i+1]  # Shape: [batch, height, width, 1]
        
        # Reshape filter for JAX's expected format: [filter_h, filter_w, in_channels, out_channels]
        f_channel = jnp.transpose(f_jax, (2, 3, 1, 0))  # Shape: [filter_h, filter_w, 1, 1]
        
        # Apply conv_transpose for single channel
        y_channel = jax.lax.conv_transpose(
            x_channel,                    # Input in NHWC format
            f_channel,                    # Filter in HWIO format
            strides=(2, 2),               # Stride of 2 in both dimensions
            padding=((f_pad + 1, f_pad + 1), (f_pad + 1, f_pad + 1)),  # Same padding as PyTorch
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        y_jax_list.append(y_channel)
    
    # Concatenate channels
    y_jax = jnp.concatenate(y_jax_list, axis=-1)
    
    # Transpose back to NCHW for comparison
    y_jax = jnp.transpose(y_jax, (0, 3, 1, 2))
    
    assert compare_tensors(y, y_jax)
    print('conv_transpose_test1 passed!')

def conv_transpose_test2():
    # Define parameters
    batch_size = 2
    in_channels = 3  # RGB image
    height, width = 8, 8  # Small test image

    # Create a simple test input: a batch with random values
    x = torch.randn(batch_size, in_channels, height, width)
    x_jax = jnp.array(x.numpy())
    # Channel last for jax convention
    x_jax = jnp.transpose(x_jax, (0, 2, 3, 1))
    print(f"Input shape: {x.shape}")
    
    # Create resample filter
    resample_filter = [1,1]
    f = torch.as_tensor(resample_filter, dtype=torch.float32)
    f = f.ger(f).unsqueeze(0).unsqueeze(1)/ f.sum().square()
    f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
    
    # PyTorch implementation
    y = torch.nn.functional.conv_transpose2d(
        x, 
        f.mul(4).tile([in_channels, 1, 1, 1]), 
        groups=in_channels, 
        stride=2, 
        padding=f_pad
    )
    
    # JAX implementation
    # Create filter in JAX
    # f_jax = jnp.array(resample_filter, dtype=jnp.float32)
    # f_jax = jnp.outer(f_jax, f_jax).reshape(1, 1, len(resample_filter), len(resample_filter)) / jnp.square(jnp.sum(f_jax))
    # f_jax = f_jax * 4  # Multiply by 4 as in PyTorch
    f_jax = jnp.array(resample_filter, dtype=jnp.float32)
    f_outer = jnp.outer(f_jax, f_jax)
    f_jax = f_outer.reshape(1, 1, *f_outer.shape) / jnp.sum(f_jax)**2
    
    assert compare_tensors(f, f_jax)

    # Reshape input and filter for grouped convolution
    # Input: [batch, height, width, channels] -> [batch*channels, height, width, 1]
    x_reshaped = x_jax.reshape(-1, height, width, 1)
    
    # Filter: [1, 1, filter_h, filter_w] -> [filter_h, filter_w, 1, 1]
    f_reshaped = jnp.transpose(f_jax, (2, 3,0, 1))
    
    # Apply conv_transpose to reshaped input
    y_reshaped = jax.lax.conv_transpose(
        x_reshaped,                    # Input in NHWC format
        f_reshaped,                    # Filter in HWIO format
        strides=(2, 2),               # Stride of 2 in both dimensions
        padding=((f_pad + 1, f_pad + 1), (f_pad + 1, f_pad + 1)),  # Same padding as PyTorch
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')  # Specify dimension order
    )
    
    # Reshape output back to original format
    # [batch*channels, out_height, out_width, 1] -> [batch, out_height, out_width, channels]
    
    y_jax = y_reshaped.reshape(batch_size, height*2, width*2, in_channels)
    
    # Transpose back to NCHW for comparison
    y_jax = jnp.transpose(y_jax, (0, 3, 1, 2))
    
    assert compare_tensors(y, y_jax, debug=True)
    print('conv_transpose_test2 passed!')

def test_positional_embedding():
    # Initialize parameters
    num_channels = 32
    batch_size = 2
    
    # Create test input
    x = torch.randn(batch_size)
    x_jax = jnp.array(x.numpy())
    
    # Create modules
    torch_pe = TorchPositionalEmbedding(num_channels=num_channels)
    flax_pe = PositionalEmbedding(num_channels=num_channels)
    
    # Forward pass
    torch_out = torch_pe(x)
    flax_out = flax_pe(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)
    print("PositionalEmbedding module test passed!")

def test_fourier_embedding():
    # Initialize parameters
    num_channels = 32
    batch_size = 2
    rngs = nnx.Rngs(0, params=1)
    
    # Create test input
    x = torch.randn(batch_size)
    x_jax = jnp.array(x.numpy())
    
    # Create modules
    torch_fe = TorchFourierEmbedding(num_channels=num_channels)
    flax_fe = FourierEmbedding(rngs=rngs, num_channels=num_channels)
    
    # Set the same frequencies for both modules for comparison
    freqs = torch_fe.freqs.detach().numpy()
    flax_fe.freqs = nnx.Param(jnp.array(freqs))
    
    # Forward pass
    torch_out = torch_fe(x)
    flax_out = flax_fe(x_jax)
    
    # Compare outputs
    assert compare_tensors(torch_out, flax_out)
    print("FourierEmbedding module test passed!")

def test_unet_block():
    # Initialize parameters
    batch_size = 2
    in_channels = 32
    out_channels = 64
    emb_channels = 16
    height = 16
    width = 16
    rngs = nnx.Rngs(0, params=1)
    
    # Create test inputs
    x = torch.randn(batch_size, in_channels, height, width)
    emb = torch.randn(batch_size, emb_channels)
    x_jax = jnp.array(x.numpy())
    # Convert from NCHW to NHWC for JAX
    x_jax = jnp.transpose(x_jax, (0, 2, 3, 1))
    emb_jax = jnp.array(emb.numpy())
    
    # Test cases with different configurations
    configs = [
        dict(attention=False, up=False, down=False),  # Basic case
        dict(attention=True, up=False, down=False),   # With attention
        dict(attention=False, up=True, down=False),   # With upsampling
        dict(attention=False, up=False, down=True),   # With downsampling
    ]
    
    for i, config in enumerate(configs):
        print(f"\nTesting UNetBlock configuration {i + 1}:", config)
        
        # Create modules with correct initialization parameters
        init_params = dict(init_mode='test')
        torch_unet = TorchUNetBlock(
            in_channels, out_channels, emb_channels,
            init=init_params,
            init_zero=dict(init_mode='test', init_weight=0),
            init_attn=init_params,
            **config
        )
        flax_unet = UNetBlock(
            rngs, in_channels, out_channels, emb_channels, 
            init=init_params, 
            init_zero=dict(init_mode='test', init_weight=0),
            init_attn=init_params,
            **config)
        
        # Forward pass
        torch_out = torch_unet(x, emb)
        flax_out = flax_unet(x_jax, emb_jax)
        # Convert back to NCHW for comparison
        flax_out = jnp.transpose(flax_out, (0, 3, 1, 2))
        
        # Compare outputs
        assert compare_tensors(torch_out, flax_out, rtol=1e-2, atol=1e-2, debug=True), f"UNetBlock {i + 1} failed"
        print(f"UNetBlock test {i + 1} passed!")

def nchw_to_nhwc(x):
    return jnp.transpose(x, (0, 2, 3, 1))

def nhwc_to_nchw(x):
    return jnp.transpose(x, (0, 3, 1, 2))

if __name__ == "__main__":
    test_linear_module()
    test_conv2d()
    test_group_norm()
    test_positional_embedding()
    test_fourier_embedding()
    test_unet_block()
    
    # Debuging methods
    # debug_conv2d_implementations()
    # temp()
    # conv_transpose_test1()
    # conv_transpose_test2()