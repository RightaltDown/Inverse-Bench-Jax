import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Callable, Dict, Optional
import optax

class EDMLossJax:
    """JAX implementation of EDM loss function."""
    
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        """Initialize EDM loss.
        
        Args:
            P_mean: Mean of log-normal distribution for noise levels
            P_std: Standard deviation of log-normal distribution for noise levels
            sigma_data: Data standard deviation for importance weights
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
    
    @jax.jit
    def __call__(self, net: nnx.Module, images: jnp.ndarray, rngs: Dict[str, Any], 
                 labels: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute EDM loss.
        
        Args:
            net: The neural network model
            images: Input images in NHWC format
            rngs: Dictionary of PRNGKeys for random operations
            labels: Optional labels for conditional generation
            
        Returns:
            Loss value as JAX array
        """
        # Generate random noise levels (log-normal distribution)
        noise_rng = rngs.get('noise', None)
        rnd_normal = jax.random.normal(noise_rng, shape=(images.shape[0], 1, 1, 1), device=images.device)
        sigma = jnp.exp(rnd_normal * self.P_std + self.P_mean)
        
        # Calculate importance weights
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        # Add noise to images
        n = jax.random.normal(noise_rng, shape=images.shape) * sigma
        noisy_images = images + n
        
        # Get model prediction
        D_yn = net(noisy_images, sigma, labels)
        
        # Calculate weighted MSE loss
        loss = weight * jnp.mean((D_yn - images) ** 2, axis=(1, 2, 3))
        
        return jnp.mean(loss) 

def mnist_loss_fn(model, batch):
    # this might run into issues
    print("Calculating loss")
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    
     # Return additional metrics in aux dict
    aux = {
        'accuracy': (logits.argmax(axis=-1) == batch['label']).mean(),
        'prediction_error': jnp.abs(logits - batch['label']).mean(),
        'logits': logits,
    }
    
    return loss, aux

class TestLoss:
    def __init__(self):
        pass
    def __call__(self, net, imgs, labels=None):
        # this might run into issues
        print("Calculating loss")
        logits = net(imgs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        
        # Return additional metrics in aux dict
        aux = {
            'accuracy': (logits.argmax(axis=-1) == imgs).mean(),
            'prediction_error': jnp.abs(logits - imgs).mean(),
            'logits': logits,
        }
        return loss, aux