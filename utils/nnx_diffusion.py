from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx

class DiffusionSampler(nnx.Module):
    """
    Diffusion sampler for reverse SDE or PF-ODE
    """
    def __init__(self, sigma_steps, factor_steps, scaling_steps, scaling_factor, num_steps, sigma_max):
        """
        Initializes the diffusion sampler with the given parameters.
        Parameters:
            sigma_steps (jnp.ndarray): Array of sigma values for each step.
            factor_steps (jnp.ndarray): Array of factor values for each step.
            scaling_steps (jnp.ndarray): Array of scaling values for each step.
            scaling_factor (jnp.ndarray): Array of scaling factors for each step.
            num_steps (int): Number of sampling steps.
            sigma_max (float): Maximum sigma value for initial noise.
        """
        super().__init__()
        self.sigma_steps = sigma_steps
        self.factor_steps = factor_steps
        self.scaling_steps = scaling_steps
        self.scaling_factor = scaling_factor
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.solver = 'euler'
    
    def score(self, model, x, sigma):
        """
        Computes the score function for the given model.
        Parameters:
            model: Diffusion model.
            x (jnp.ndarray): Input tensor.
            sigma (float): Sigma value.
        Returns:
            jnp.ndarray: The computed score.
        """
        sigma = jnp.asarray(sigma)
        d = model(x, sigma)
        return (d - x) / sigma**2
    
    def euler_step(self, model, x, step_idx, rng, SDE=False):
        """
        Single step of Euler's method for sampling.
        """
        sigma = self.sigma_steps[step_idx]
        factor = self.factor_steps[step_idx]
        scaling_factor = self.scaling_factor[step_idx]
        
        score = self.score(model, x / self.scaling_steps[step_idx], sigma) / self.scaling_steps[step_idx]
        
        if SDE:
            noise_rng, rng = jax.random.split(rng)
            epsilon = jax.random.normal(noise_rng, x.shape)
            x = x * scaling_factor + factor * score + jnp.sqrt(factor) * epsilon
        else:
            x = x * scaling_factor + factor * score * 0.5
            
        return x, rng
    
    # New
    @partial(nnx.jit, static_argnums=(0,))
    def sample_step(self, carry, step_idx):
        """
        Single step of the sampling process using Euler method.
        """
        x, rng = carry
        rng, step_rng = jax.random.split(rng)
        
        sigma = self.sigma_steps[step_idx]
        factor = self.factor_steps[step_idx]
        
        score = self.score(self.model, x, sigma)
        
        # Deterministic ODE step (non-SDE version)
        x_new = x + factor * score * 0.5
        
        return (x_new, rng), x_new
    
    def sample(self, model, rng, shape, SDE=False, verbose=False):
        """
        Samples from the diffusion process using the specified model.
        Parameters:
            model: Diffusion model supports 'score' and 'tweedie'
            rng: JAX random number generator key
            shape (tuple): Shape of the output sample
            SDE (bool): Whether to use Stochastic Differential Equations.
            verbose (bool): Whether to display progress bar.
        Returns:
            jnp.ndarray: The final sampled state.
        """
        # Generate initial noise
        noise_rng, step_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, shape) * self.sigma_max
        
        # Define a step function that can be called in a loop
        if verbose:
            for step in tqdm(range(self.num_steps)):
                x, step_rng = self.euler_step(model, x, step, step_rng, SDE)
        else:
            # For JAX efficiency, we could use scan instead of a Python loop
            def scan_fn(carry, step_idx):
                x, rng = carry
                x_new, rng_new = self.euler_step(model, x, step_idx, rng, SDE)
                return (x_new, rng_new), None
            
            indices = jnp.arange(self.num_steps)
            (x, step_rng), _ = jax.lax.scan(scan_fn, (x, step_rng), indices)
        
        return x
    
    def get_start(self, rng, shape):
        """
        Generates a random initial state.
        Parameters:
            rng: JAX random number generator key
            shape (tuple): Shape of the output sample
        Returns:
            jnp.ndarray: Initial random state.
        """
        return jax.random.normal(rng, shape) * self.sigma_max