import os
import sys
import torch
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from navier_stokes import NavierStokes2d, ForwardNavierStokes2d  # Import torch version
from nnx_navier_stokes import NavierStokes2dJAX, ForwardNavierStokes2dJAX  # Import jax version

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def compare_tensors(torch_tensor, jax_tensor, rtol=1e-5, atol=1e-5):
    """Compare PyTorch and JAX tensors with given tolerances"""
    torch_tensor = torch_tensor.detach().numpy()
    jax_tensor = np.array(jax_tensor)
    return np.allclose(torch_tensor, jax_tensor, rtol=rtol, atol=atol)

def save_visualization(torch_result, jax_result, diff, max_diff, filename):
    """Helper function to save visualization of differences between implementations"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].imshow(torch_result, cmap='RdYlBu')
    axes[0].set_title('PyTorch Result')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(jax_result, cmap='RdYlBu')
    axes[1].set_title('JAX Result')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(diff, cmap='viridis')
    axes[2].set_title(f'Absolute Difference (Max: {max_diff:.6f})')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def test_navier_stokes_solver():
    """Test that the PyTorch and JAX implementations of the Navier-Stokes solver produce equivalent results"""
    # Test parameters
    resolution = 64  # Lower resolution for testing
    L = 2 * np.pi
    Re = 200.0
    forward_time = 1.0
    delta_t = 1e-2
    adaptive = True
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    jax_key = jax.random.PRNGKey(42)
    
    # Initialize solvers
    torch_solver = NavierStokes2d(resolution, resolution, L, L, Re=Re, device=torch.device('cpu'), dtype=torch.float32)
    jax_solver = NavierStokes2dJAX(resolution, resolution, L, L, Re=Re, dtype=jnp.float32)
    
    # Get the forcing term
    torch_force = -4 * torch.cos(4.0 * torch.linspace(0, L, resolution+1)[:-1].reshape(-1, 1).repeat(1, resolution))
    jax_force = -4 * jnp.cos(4.0 * jnp.linspace(0, L, resolution+1)[:-1].reshape(-1, 1).repeat(resolution, axis=1))
    
    # Create random initial vorticity field
    torch_w_init = torch.randn(resolution, resolution, dtype=torch.float32)
    jax_w_init = jnp.array(torch_w_init.numpy())  # Convert to JAX array
    
    # Solve using both implementations
    torch_w_final = torch_solver.solve(
        torch_w_init, 
        torch_force, 
        T=forward_time, 
        Re=Re, 
        adaptive=adaptive, 
        delta_t=delta_t
    )
    
    jax_w_final = jax_solver.solve(
        jax_w_init, 
        jax_force, 
        T=forward_time, 
        Re=Re, 
        adaptive=adaptive, 
        delta_t=delta_t
    )
    
    # Convert results to numpy for comparison
    torch_result_np = torch_w_final.numpy()
    jax_result_np = np.array(jax_w_final)
    
    # Calculate difference
    diff = np.abs(jax_result_np - torch_result_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Optional: Save visualization for debugging if difference is significant
    if max_diff >= 1e-5:
        save_visualization(
            torch_result_np, 
            jax_result_np, 
            diff, 
            max_diff, 
            "navier_stokes_comparison.png"
        )
    
    # Assert that the implementations are equivalent within a tolerance
    assert max_diff < 1e-4, f"Maximum difference ({max_diff}) exceeds tolerance"
    assert mean_diff < 1e-5, f"Mean difference ({mean_diff}) exceeds tolerance"

def test_forward_operator():
    """Test that the PyTorch and JAX implementations of the forward operator produce equivalent results"""
    # Test parameters
    resolution = 64
    L = 2 * np.pi
    Re = 200.0
    forward_time = 1.0
    delta_t = 1e-2
    adaptive = True
    batch_size = 2
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    jax_key = jax.random.PRNGKey(42)
    
    # Initialize forward operators
    torch_forward = ForwardNavierStokes2d(
        resolution=resolution, 
        L=L, 
        forward_time=forward_time,
        Re=Re, 
        delta_t=delta_t, 
        adaptive=adaptive, 
        device='cpu'
    )
    
    jax_forward = ForwardNavierStokes2dJAX(
        resolution=resolution, 
        L=L, 
        forward_time=forward_time,
        Re=Re, 
        delta_t=delta_t, 
        adaptive=adaptive
    )
    
    # Create a batch of input data
    torch_input = torch.randn(batch_size, 1, resolution, resolution, dtype=torch.float32)
    jax_input = jnp.array(torch_input.numpy())
    
    # Create data dictionaries
    torch_data = {'target': torch_input}
    jax_data = {'target': jax_input}
    
    # Run forward operators
    torch_output = torch_forward(torch_data, unnormalize=False)
    jax_output = jax_forward(jax_data, unnormalize=False, key=jax_key)
    
    # Convert results to numpy for comparison
    torch_output_np = torch_output.numpy()
    jax_output_np = np.array(jax_output)
    
    # Calculate difference
    forward_diff = np.abs(jax_output_np - torch_output_np)
    forward_max_diff = np.max(forward_diff)
    forward_mean_diff = np.mean(forward_diff)
    
    # Optional: Save visualization for debugging if difference is significant
    if forward_max_diff >= 1e-5:
        save_visualization(
            torch_output_np[0, 0], 
            jax_output_np[0, 0], 
            forward_diff[0, 0], 
            forward_max_diff, 
            "navier_stokes_forward_comparison.png"
        )
    
    # Assert that the implementations are equivalent within a tolerance
    assert forward_max_diff < 1e-4, f"Forward op maximum difference ({forward_max_diff}) exceeds tolerance"
    assert forward_mean_diff < 1e-5, f"Forward op mean difference ({forward_mean_diff}) exceeds tolerance"

if __name__ == '__main__':
    pytest.main([__file__])