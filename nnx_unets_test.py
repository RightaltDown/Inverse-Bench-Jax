import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx

# Import the JAX/Flax implementation
from models.nnx_unets import DhariwalUNet
from models.unets import DhariwalUNet as torchUnet

def compare_tensors(torch_tensor, jax_tensor, rtol=1e-3, atol=1e-3, debug=False):
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

def test_dhariwal_unet():
    """
    Test the Flax nnx implementation of DhariwalUNet against the PyTorch version.
    """
    print("Testing DhariwalUNet implementation...")
    
    img_resolution=  32
    in_channels =  1
    out_channels = 1
    label_dim = 0
    model_channels =  128
    channel_mult =  [1, 1, 1, 2, 2]
    attn_resolutions =  [16]
    num_blocks = 1
    dropout = 0.0
    
    # Create random input data
    batch_size = 2
    
    # JAX implementation with fixed seed
    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, 5)
    
    x_jax = jax.random.normal(subkeys[0], (batch_size, img_resolution, img_resolution, in_channels))
    noise_labels_jax = jax.random.normal(subkeys[1], (batch_size,))
    # class_labels_jax = jax.nn.one_hot(jax.random.randint(subkeys[2], (batch_size,), 0, label_dim), label_dim)
    class_labels_jax = None
    
    # Initialize JAX model
    model_key = subkeys[3]
    params_key, dropout_key = jax.random.split(model_key)
    
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    model_jax = DhariwalUNet(
        rngs=rngs,
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
    )
    
    # Forward pass in JAX
    output_jax = model_jax(x_jax, noise_labels_jax, class_labels_jax, train=False)
    
    print(f"JAX output shape: {output_jax.shape}")
    
    # Convert to PyTorch tensors for comparison with PyTorch implementation
    # This part would be used to compare with the actual PyTorch implementation
    # For this example, we just show how you would set up the comparison
    x_torch = torch.from_numpy(np.array(x_jax)).permute(0, 3, 1, 2)  # NHWC to NCHW
    noise_labels_torch = torch.from_numpy(np.array(noise_labels_jax))
    class_labels_torch = torch.from_numpy(np.array(class_labels_jax))
    
    print("\nSetup for PyTorch comparison:")
    print(f"Torch input shape: {x_torch.shape}")
    # print(f"Torch noise labels shape: {noise_labels_torch.shape}")
    # print(f"Torch class labels shape: {class_labels_torch.shape}")
    
    model_torch = torchUnet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
    )
    
    output_torch = model_torch(x_torch, noise_labels_torch, class_labels_torch)
    
    print(f"torch output shape: {output_torch.shape}")
    
    output_jax = jnp.transpose(output_jax, (0, 3, 1, 2))
    assert compare_tensors(output_torch, output_jax, debug=True)
    print("DhariwalUnet test passed!")
    
if __name__ == "__main__":
    # Run the tests
    test_dhariwal_unet()