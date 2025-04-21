from functools import partial
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm.auto import trange

class DiffusionSampler:
    """
    Diffusion sampler for reverse SDE or PF-ODE (JAX/NNX implementation)
    """

    def __init__(self, scheduler, solver='euler'):
        """
        Initializes the diffusion sampler with the given scheduler and solver.

        Parameters:
            scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
            solver (str): Solver method ('euler').
        """
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, rngs=None, SDE=False, verbose=False):
        """
        Samples from the diffusion process using the specified model.

        Parameters:
            model: Diffusion model that supports 'score' and 'tweedie'
            x_start (jnp.ndarray): Initial state.
            key (jax.random.PRNGKey): JAX random key for stochastic sampling.
            SDE (bool): Whether to use Stochastic Differential Equations.
            verbose (bool): Whether to display progress bar.

        Returns:
            jnp.ndarray: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, rngs, SDE, verbose)
        else:
            raise NotImplementedError

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
        sigma = jnp.asarray(sigma, dtype=x.dtype)
        d = model(x, sigma)
        return (d - x) / (sigma ** 2)
    
    def _euler_step(self, i, state, model, SDE, random_seq=None):
        """
        Single Euler integration step.
        
        Parameters:
            i (int): Current step index.
            state (jnp.ndarray): Current state.
            model: Diffusion model.
            SDE (bool): Whether to add stochastic noise.
            random_seq (jnp.ndarray, optional): Pre-generated random noise sequence.
            
        Returns:
            jnp.ndarray: Updated state.
        """
        x = state
        sigma = self.scheduler.sigma_steps[i]
        factor = self.scheduler.factor_steps[i]
        scaling_factor = self.scheduler.scaling_factor[i]
        
        # Calculate score
        score = self.score(model, x / self.scheduler.scaling_steps[i], sigma) / self.scheduler.scaling_steps[i]
        
        # Update state
        if SDE:
            if random_seq is not None:
                # Use pre-generated noise
                epsilon = random_seq[i]
            else:
                # This would typically be moved outside for pure functional implementation
                epsilon = jax.random.normal(jax.random.PRNGKey(i), shape=x.shape, dtype=x.dtype)
            
            x = x * scaling_factor + factor * score + jnp.sqrt(factor) * epsilon
        else:
            x = x * scaling_factor + factor * score * 0.5
            
        return x
    
    def _euler(self, model, x_start, rngs, SDE=False, verbose=False):
        """
        Euler's method for sampling from the diffusion process.
        
        This implementation offers both a standard loop with tqdm for visualization
        and a pure functional implementation using jax.lax.scan.
        """
        num_steps = self.scheduler.num_steps
        
        # Pre-generate all noise if using SDE for reproducibility
        random_seq = None
        if SDE and rngs is not None:
            # Generate all random noise upfront
            subkeys = jax.random.split(rngs.sampling(), num_steps)
            random_seq = jax.vmap(lambda k: jax.random.normal(k, shape=x_start.shape, dtype=x_start.dtype))(subkeys)
        
        # Use tqdm for progress tracking if verbose
        if verbose:
            x = x_start
            for i in trange(num_steps):
                x = self._euler_step(i, x, model, SDE, random_seq[i] if random_seq is not None else None)
            return x
        else:
            # Pure functional implementation using scan
            def scan_step(x, i):
                new_x = self._euler_step(i, x, model, SDE, random_seq[i] if random_seq is not None else None)
                return new_x, None
            
            final_x, _ = jax.lax.scan(scan_step, x_start, jnp.arange(num_steps))
            return final_x

    def get_start(self, ref_shape, rngs, dtype=jnp.float32):
        """
        Generates a random initial state based on the reference shape.

        Parameters:
            ref_shape (tuple): Shape for the random tensor.
            key (jax.random.PRNGKey): Random key for generation.
            dtype: Data type for the generated tensor.

        Returns:
            jnp.ndarray: Initial random state.
        """
        return jax.random.normal(rngs.sampling(), shape=ref_shape, dtype=dtype) * self.scheduler.sigma_max