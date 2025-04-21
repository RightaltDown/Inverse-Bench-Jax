import jax
import jax.numpy as jnp
import nnx
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import nnx_diffusion as DiffusionSampler

class UnconditionalDiffusionSampler(Algo):
    def __init__(self, net, forward_op, diffusion_scheduler_config={}, sde=False):
        super(UnconditionalDiffusionSampler, self).__init__(net, forward_op)
        self.net = net
        # Note: No eval() or requires_grad_(False) in JAX/NNX
        # We'll handle this differently with function transformation
        self.forward_op = forward_op
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.sde = sde
        
    def inference(self, rngs, observation, key=None, num_samples=1, verbose=True):
        """
        Generate samples using the diffusion model
        
        Args:
            observation: Input observation tensor
            key: JAX random key for random number generation
            num_samples: Number of samples to generate
            verbose: Whether to show progress bar
            
        Returns:
            The generated samples
        """
        # In JAX, we need to explicitly pass the random key
        if rngs is None:
            rngs = nnx.Rngs(sampling=0)
            
        # Create scheduler
        diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config)
        
        # Create initial noise
        noise_shape = (observation.shape[0], self.net.img_resolution,self.net.img_resolution, self.net.img_channels)
        xt = jax.random.normal(rngs.sampling(), shape=noise_shape) * diffusion_scheduler.sigma_max
        
        sampler = DiffusionSampler(diffusion_scheduler)
        xt = sampler.sample(self.net, xt, rngs=rngs, SDE=self.sde, verbose=False)
        
        return xt