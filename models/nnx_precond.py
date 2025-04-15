from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from models.nnx_unets import DhariwalUNet

_model_dict = {
    'DhariwalUNet': DhariwalUNet
}

class EDMPrecond(nnx.Module):
    def __init__(self, rngs,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        self.model = _model_dict[model_type](
            rngs,
            img_resolution=img_resolution, 
            in_channels=img_channels, 
            out_channels=img_channels, 
            label_dim=label_dim, 
            **model_kwargs
        )
    
    # train is usually False? 
    def __call__(self, x, sigma, class_labels=None, force_fp32=False, train=True, **model_kwargs):
        # Convert inputs if needed (in PyTorch this would convert to Tensor, in JAX we ensure array type)
        # Reshape sigma to match PyTorch's reshape(-1, 1, 1, 1)
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        sigma = sigma.reshape(-1, 1, 1, 1)
        
        # Handle class labels - similar to PyTorch implementation
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = jnp.zeros((1, self.label_dim), dtype=jnp.float32)
        else:
            class_labels = jnp.asarray(class_labels, dtype=jnp.float32).reshape(-1, self.label_dim)
        
        # Calculate conditioning parameters
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4
        
        # Determine precision
        # In JAX, we'd handle this differently than PyTorch's dtype switching
        dtype = jnp.float32
        if self.use_fp16 and not force_fp32:
            # Note: In practice, you'd want to check if JAX is configured to use GPU
            # and if the operation supports f16. This is a simplification.
            dtype = jnp.float16
        
        # Scale input and run the model
        x_scaled = c_in * x
        if dtype == jnp.float16:
            x_scaled = x_scaled.astype(dtype)
        
        F_x = self.model(
            x_scaled, 
            jnp.ravel(c_noise),  # flatten() in PyTorch = ravel() in JAX
            class_labels=class_labels, 
            train=train,
            **model_kwargs
        )
        
        # Apply skip connection and scaling - convert back to float32 if needed
        if F_x.dtype != jnp.float32:
            F_x = F_x.astype(jnp.float32)
        
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def round_sigma(self, sigma):
        """Convert sigma to the appropriate format."""
        return jnp.asarray(sigma)

_precond_dict = {
    'edm': EDMPrecond
}


def get_model(name, **kwargs):
    return _precond_dict[name.lower()](**kwargs)