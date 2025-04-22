import math
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Dict, Tuple, Optional, Any
# from .base import BaseOperator
from base import BaseOperator

class NavierStokes2dJAX(object):
    """
    JAX implementation of the pseudo-spectral solver for 2D Navier-Stokes equation
    """
    def __init__(self, s1, s2, 
                 L1=2*math.pi, L2=2*math.pi,
                 Re=100.0, dtype=jnp.float32):
        """
        Args:
            - s1, s2: spatial resolution
            - L1, L2: spatial domain
            - Re: Reynolds number
            - dtype: data type
        """
        self.s1 = s1
        self.s2 = s2
        
        self.L1 = L1
        self.L2 = L2
        
        self.Re = Re
        
        self.h = 1.0/max(s1, s2)
        self.dtype = dtype

        # Wavenumbers for first derivatives
        freq_list1 = jnp.concatenate([
            jnp.arange(0, s1//2, 1),
            jnp.zeros(1),
            jnp.arange(-s1//2 + 1, 0, 1)
        ])
        self.k1 = jnp.tile(freq_list1[:, jnp.newaxis], (1, s2//2 + 1)).astype(dtype)

        freq_list2 = jnp.concatenate([jnp.arange(0, s2//2, 1), jnp.zeros(1)])
        self.k2 = jnp.tile(freq_list2[jnp.newaxis, :], (s1, 1)).astype(dtype)

        # Negative Laplacian
        freq_list1 = jnp.concatenate([jnp.arange(0, s1//2, 1), jnp.arange(-s1//2, 0, 1)])
        k1 = jnp.tile(freq_list1[:, jnp.newaxis], (1, s2//2 + 1)).astype(dtype)

        freq_list2 = jnp.arange(0, s2//2 + 1, 1)
        k2 = jnp.tile(freq_list2[jnp.newaxis, :], (s1, 1)).astype(dtype)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        # Inverse of negative Laplacian
        self.inv_lap = self.G.copy()
        # Replace the first element with 1.0 to avoid division by zero
        self.inv_lap = self.inv_lap.at[0, 0].set(1.0)
        self.inv_lap = 1.0/self.inv_lap

        # Dealiasing mask using 2/3 rule
        self.dealias = (k1**2 + k2**2 <= (s1/3)**2 + (s2/3)**2).astype(dtype)
        # Ensure mean zero
        self.dealias = self.dealias.at[0, 0].set(0.0)

    def stream_function(self, w_h, real_space=False):
        """
        Compute stream function from vorticity (Fourier space)
        -Lap(psi) = w
        """
        psi_h = self.inv_lap * w_h

        if real_space:
            return jnp.fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    def velocity_field(self, stream_f, real_space=True):
        """
        Compute velocity field from stream function (Fourier space)
        """
        # Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2) * 1j * self.k2 * stream_f

        # Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1) * 1j * self.k1 * stream_f

        if real_space:
            return (jnp.fft.irfft2(q_h, s=(self.s1, self.s2)), 
                    jnp.fft.irfft2(v_h, s=(self.s1, self.s2)))
        else:
            return q_h, v_h

    def nonlinear_term(self, w_h, f_h=None):
        """
        Compute non-linear term + forcing from given vorticity (Fourier space)
        """
        # Dealias vorticity
        dealias_w_h = w_h * self.dealias

        # Physical space vorticity
        w = jnp.fft.irfft2(dealias_w_h, s=(self.s1, self.s2))

        # Physical space velocity
        q, v = self.velocity_field(self.stream_function(dealias_w_h, real_space=False), real_space=True)

        # Compute non-linear term in Fourier space
        nonlin = -1j * ((2*math.pi/self.L1) * self.k1 * jnp.fft.rfft2(q*w) + 
                         (2*math.pi/self.L1) * self.k2 * jnp.fft.rfft2(v*w))

        # Add forcing function
        if f_h is not None:
            nonlin = nonlin + f_h

        return nonlin

    def time_step(self, q, v, f, Re):
        """Calculate the appropriate time step based on CFL condition"""
        # Maximum speed
        max_speed = jnp.max(jnp.sqrt(q**2 + v**2))

        # Maximum force amplitude
        if f is not None:
            xi = jnp.sqrt(jnp.max(jnp.abs(f)))
        else:
            xi = 1.0
        
        # Viscosity
        mu = (1.0/Re) * xi * ((self.L1/(2*math.pi))**(3.0/4.0)) * (((self.L2/(2*math.pi))**(3.0/4.0)))

        # Compute adaptive time step
        cfl_dt = 0.5 * self.h / jnp.maximum(max_speed, 1e-10)  # Avoid division by zero
        visc_dt = 0.5 * (self.h**2) / mu
        
        return jnp.minimum(cfl_dt, visc_dt)

    def solve_step(self, w_h, f_h, GG, current_delta_t):
        """Single step solver update for Navier-Stokes"""
        # Inner-step of Heun's method
        nonlin1 = self.nonlinear_term(w_h, f_h)
        w_h_tilde = (w_h + current_delta_t * (nonlin1 - 0.5 * GG * w_h)) / (1.0 + 0.5 * current_delta_t * GG)

        # Cranck-Nicholson + Heun update
        nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
        w_h_new = (w_h + current_delta_t * (0.5 * (nonlin1 + nonlin2) - 0.5 * GG * w_h)) / (1.0 + 0.5 * current_delta_t * GG)
        
        return w_h_new

    def solve(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        """
        Solve 2D Navier-Stokes equation
        
        Args:
            w: Initial vorticity field
            f: Forcing function (optional)
            T: Total integration time
            Re: Reynolds number
            adaptive: Whether to use adaptive time stepping
            delta_t: Time step (used if adaptive=False)
            
        Returns:
            Final vorticity field
        """
        # Rescale Laplacian by Reynolds number
        GG = (1.0/Re) * self.G

        # Move to Fourier space
        w_h = jnp.fft.rfft2(w)

        if f is not None:
            f_h = jnp.fft.rfft2(f)
        else:
            f_h = None
            
        # Initial time step if adaptive
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re)

        # State for the loop
        init_state = (0.0, w_h, delta_t)
        
        # Define the condition for the while loop
        def cond_fun(state):
            time, _, _ = state
            return time < T
            
        # Define the body of the while loop
        def body_fun(state):
            time, w_h, dt = state
            
            # Calculate current delta_t
            current_delta_t = jnp.minimum(dt, T - time)
            
            # Update w_h
            w_h_new = self.solve_step(w_h, f_h, GG, current_delta_t)
            
            # Update time
            new_time = time + current_delta_t
            
            # Calculate new time step if adaptive
            new_dt = lax.cond(
                adaptive,
                lambda _: self.time_step(*self.velocity_field(self.stream_function(w_h_new, real_space=False), real_space=True), f, Re),
                lambda _: dt,
                operand=None
            )
            
            return (new_time, w_h_new, new_dt)
            
        # Use while_loop to advance the solution
        _, final_w_h, _ = lax.while_loop(cond_fun, body_fun, init_state)
        
        # Convert back to real space
        return jnp.fft.irfft2(final_w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.solve(w, f, T, Re, adaptive, delta_t)


class ForwardNavierStokes2dJAX(BaseOperator):
    """
    JAX implementation of Forward operator for 2D Navier-Stokes equation
    """
    def __init__(self, 
                 resolution=128, L=2 * math.pi,
                 forward_time=1.0,
                 Re=200.0, 
                 downsample_factor=2,
                 dtype=jnp.float32,
                 delta_t=1e-2,
                 adaptive=True,
                 sigma_noise=0.0):
        
        self.dtype = dtype
        self.solver = NavierStokes2dJAX(resolution, resolution, L, L, dtype=dtype)
        self.force = self.get_forcing(resolution, L)

        self.downsample_factor = downsample_factor
        self.forward_time = forward_time
        self.Re = Re
        self.delta_t = delta_t
        self.adaptive = adaptive
        self.sigma_noise = sigma_noise
        self.key = jax.random.PRNGKey(42)  # Default key

    def get_forcing(self, resolution, L):
        """Generate the forcing term"""
        t = jnp.linspace(0, L, resolution+1)[0:-1]
        _, y = jnp.meshgrid(t, t, indexing='ij')
        return -4 * jnp.cos(4.0 * y)

    def unnormalize(self, x):
        """Identity function as placeholder - implement actual unnormalization if needed"""
        return x

    def forward(self, x, unnormalize=True, key=None):
        """
        Forward operator for Navier-Stokes
        
        Args:
            x: velocity field of shape (batch_size, 1, resolution, resolution)
            unnormalize: whether to unnormalize the input data
            key: JAX random key for noise
            
        Returns:
            solution velocity field
        """
        # Unnormalize if requested
        if unnormalize:
            raw_u = self.unnormalize(x)
        else:
            raw_u = x

        # Remove channel dimension for solver
        raw_u = jnp.squeeze(raw_u, axis=1)
        
        # Solve the PDE
        # We need to vmap over the batch dimension
        solve_batch = jax.vmap(self.solver, in_axes=(0, None, None, None, None, None))
        sol = solve_batch(
            raw_u, 
            self.force, 
            self.forward_time, 
            self.Re, 
            self.adaptive, 
            self.delta_t
        )
        
        # Downsample the velocity field
        sol = sol[:, ::self.downsample_factor, ::self.downsample_factor]
        
        # Add channel dimension back
        sol = sol[:, jnp.newaxis, :, :]
        
        # Generate and add noise if sigma_noise > 0
        if self.sigma_noise > 0 and key is not None:
            noise = jax.random.normal(key, sol.shape) * self.sigma_noise
            sol = sol + noise
            
        return sol.astype(jnp.float32)

    def __call__(self, data, unnormalize=True, key=None):
        """
        Call method for compatibility with the original PyTorch version
        
        Args:
            data: Dictionary containing the initial vorticity field with key 'target'
            unnormalize: Whether to unnormalize the input data
            key: JAX random key for noise
            
        Returns:
            Solution velocity field
        """
        # If key is not provided, use the internal key and update it
        if key is None:
            key = self.key
            _, self.key = jax.random.split(key)
            
        # Extract target from data dictionary
        x = data['target']
        
        # Call forward method
        sol = self.forward(x, unnormalize, key)
        
        return sol