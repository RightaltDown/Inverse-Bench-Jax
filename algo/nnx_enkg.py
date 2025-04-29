import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from flax import nnx
from tqdm.auto import tqdm
from typing import Dict, Tuple, Optional, Any, Callable


def ode_sampler(
    net: Callable,
    x_initial: jnp.ndarray,
    num_steps: int = 18,
    sigma_start: float = 80.0,
    sigma_eps: float = 0.002,
    rho: int = 7,
):
    """
    Generate x_0 from x_t for any t (JAX implementation)
    """
    # If only one step, just denoise directly
    if num_steps == 1:
        denoised = net(x_initial, sigma_start)
        return denoised
    
    last_sigma = sigma_eps
    # Time step discretization
    step_indices = jnp.arange(num_steps, dtype=jnp.float32)

    t_steps = (
        sigma_start ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (last_sigma ** (1 / rho) - sigma_start ** (1 / rho))
    ) ** rho
    
    # Round sigma if the net has this method, otherwise use directly
    # This is a placeholder for compatibility with PyTorch implementation
    if hasattr(net, 'round_sigma'):
        rounded_t_steps = jax.vmap(net.round_sigma)(t_steps)
    else:
        rounded_t_steps = t_steps
        
    # Add t_N = 0
    t_steps = jnp.concatenate([rounded_t_steps, jnp.zeros_like(t_steps[:1])])

    # Define a single step of the ODE solver
    def step_fn(x_cur, ts):
        t_cur, t_next = ts
        
        t_hat = t_cur
        x_hat = x_cur
        
        # Euler step
        denoised = net(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        return x_next, None  # Return updated x and dummy aux info

    # Use scan to iterate through all timesteps
    x_next, _ = lax.scan(step_fn, x_initial, (t_steps[:-1], t_steps[1:]))
    
    return x_next


class EnKGJAX:
    """
    Ensemble Kalman Guidance implementation in JAX
    """
    def __init__(self, 
                 net,
                 forward_op,
                 guidance_scale, 
                 num_steps, 
                 num_updates, 
                 sigma_max,
                 sigma_min,
                 num_samples=1024,
                 threshold=0.1,
                 batch_size=128,
                 lr_min_ratio=0.0,
                 rho=7, 
                 factor=4):
        """
        Initialize the EnKG algorithm
        
        Args:
            net: The diffusion model network
            forward_op: The forward operator for simulating physics
            guidance_scale: Scale factor for the guidance
            num_steps: Number of diffusion steps
            num_updates: Number of ensemble updates per step
            sigma_max: Maximum noise level
            sigma_min: Minimum noise level
            num_samples: Number of ensemble particles
            threshold: Threshold for when to apply guidance
            batch_size: Batch size for processing particles
            lr_min_ratio: Minimum learning rate ratio
            rho: Noise schedule parameter
            factor: Factor for sub-steps in updates
        """
        self.net = net
        self.forward_op = forward_op
        self.rho = rho
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.threshold = threshold
        self.num_samples = num_samples
        self.lr_min_ratio = lr_min_ratio
        self.factor = factor
    
    def get_lr(self, i):
        """Get learning rate based on current step"""
        if self.lr_min_ratio > 0.0:
            return self.guidance_scale * (1 - self.lr_min_ratio) * (self.num_steps - i) / self.num_steps + self.lr_min_ratio
        else:
            return self.guidance_scale
    
    def update_particles(self, particles, observation, num_steps, sigma_start, guidance_scale=1.0, key=None):
        """
        Update ensemble particles based on observation
        
        Args:
            particles: Ensemble of particles to update
            observation: Target observation
            num_steps: Number of steps for ODE sampler
            sigma_start: Starting sigma for ODE sampler
            guidance_scale: Scale factor for updates
            key: JAX random key
            
        Returns:
            updated particles and updated timestep
        """
        N, *spatial = particles.shape
        t_hat = sigma_start
        
        # Get batch count
        num_batches = (particles.shape[0] + self.batch_size - 1) // self.batch_size
        
        # Function to process a single batch for ode_sampler
        def process_batch(batch_idx, x0s):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, particles.shape[0])
            batch = jax.lax.dynamic_slice(particles, (start_idx,) + (0,) * len(spatial), 
                                          (end_idx - start_idx,) + tuple(spatial))
            
            # Apply ode_sampler to the batch
            batch_result = ode_sampler(
                self.net,
                batch,
                num_steps=num_steps,
                sigma_start=sigma_start,
            )
            
            # Update the corresponding slice in x0s
            return jax.lax.dynamic_update_slice(x0s, batch_result, (start_idx,) + (0,) * len(spatial))
            
        # Function to perform one update
        def perform_update(j, particle_state):
            particles, logging_info = particle_state
            
            # Initialize x0s for this update
            x0s = jnp.zeros_like(particles)
            
            # Process all batches
            x0s = jax.lax.fori_loop(0, num_batches, process_batch, x0s)
            
            # Get measurement for each particle using the forward operator
            ys = jax.vmap(self.forward_op.forward, in_axes=(0, False, None))(
                x0s[:, jnp.newaxis], False, None
            )
            # Remove extra dimensions if needed
            ys = jnp.squeeze(ys, axis=1)
            
            # Calculate ensemble means
            particles_mean = jnp.mean(particles, axis=0, keepdims=True)
            ys_mean = jnp.mean(ys, axis=0, keepdims=True)
            
            # Calculate differences from means
            xs_diff = particles - particles_mean
            ys_diff = ys - ys_mean
            
            # Calculate measurement gradient
            ys_err = 0.5 * jax.vmap(self.forward_op.gradient_m, in_axes=(0, None))(ys, observation)
            
            # Reshape for matrix operations
            xs_diff_flat = xs_diff.reshape(xs_diff.shape[0], -1)
            ys_diff_flat = ys_diff.reshape(ys_diff.shape[0], -1)
            ys_err_flat = ys_err.reshape(ys_err.shape[0], -1)
            
            # Calculate coefficient matrix
            coef = jnp.matmul(ys_err_flat, ys_diff_flat.T) / particles.shape[0]
            
            # Calculate particle updates
            dxs = jnp.matmul(coef, xs_diff_flat)
            
            # Calculate learning rate
            lr = guidance_scale / jnp.linalg.norm(coef)
            
            # Update particles
            new_particles = particles - lr * dxs.reshape(N, *spatial)
            
            # Calculate logging information
            abs_ys = jnp.abs(ys_err)
            abs_err = jnp.mean(abs_ys)
            max_err = jnp.max(abs_ys)
            std = jnp.std(particles, axis=0, keepdims=True)
            avg_std = jnp.mean(std)
            avg_norm = jnp.mean(jnp.linalg.norm(dxs, axis=1))
            
            # # Update logging info
            # new_logging_info = {
            #     "abs_error": abs_err,
            #     "max_error": max_err,
            #     "avg_update_norm": avg