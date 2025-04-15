import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Optional, Tuple

class ResBlock(nnx.Module):
    """Residual block with optional attention."""
    
    def __init__(self, channels: int, dropout: float = 0.0, use_attention: bool = False):
        super().__init__()
        self.norm1 = nnx.GroupNorm(num_groups=32, num_features=channels)
        self.conv1 = nnx.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nnx.GroupNorm(num_groups=32, num_features=channels)
        self.conv2 = nnx.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nnx.Dropout(dropout)
        
        if use_attention:
            self.attention = SelfAttention(channels)
        else:
            self.attention = None
            
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        h = self.norm1(x)
        h = jax.nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = jax.nn.silu(h)
        h = self.dropout(h, deterministic=not training)
        h = self.conv2(h)
        
        if self.attention is not None:
            h = self.attention(h)
            
        return x + h

class SelfAttention(nnx.Module):
    """Multi-head self attention layer."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nnx.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nnx.Conv2d(channels, channels, kernel_size=1)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        q, k, v = jnp.split(qkv, indices_or_sections=3, axis=2)
        q = q.squeeze(2)  # (B, H*W, num_heads, C//num_heads)
        k = k.squeeze(2)
        v = v.squeeze(2)
        
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * (C // self.num_heads) ** -0.5
        attn = jax.nn.softmax(attn, axis=-1)
        
        h = (attn @ v).reshape(B, H, W, C)
        h = self.proj(h)
        return x + h

class UNet(nnx.Module):
    """U-Net model for diffusion."""
    
    def __init__(self,
                 img_resolution: int,
                 img_channels: int,
                 model_channels: int,
                 channel_mult: List[int],
                 attn_resolutions: List[int],
                 num_blocks: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.model_channels = model_channels
        
        # Initial convolution
        self.conv_in = nnx.Conv2d(img_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down_blocks = []
        channels = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_blocks):
                layers = []
                layers.append(ResBlock(
                    channels,
                    dropout=dropout,
                    use_attention=(img_resolution // (2**level) in attn_resolutions)
                ))
                channels = model_channels * mult
                layers.append(nnx.Conv2d(channels, channels, kernel_size=3, padding=1))
                self.down_blocks.append(nnx.Sequential(*layers))
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nnx.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
        
        # Middle
        self.middle_blocks = []
        for _ in range(num_blocks):
            self.middle_blocks.append(ResBlock(channels, dropout=dropout, use_attention=True))
        
        # Upsampling
        self.up_blocks = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_blocks + 1):
                layers = []
                if level != len(channel_mult) - 1 and i == 0:
                    layers.append(nnx.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1))
                
                channels = model_channels * mult
                layers.append(ResBlock(
                    channels,
                    dropout=dropout,
                    use_attention=(img_resolution // (2**level) in attn_resolutions)
                ))
                
                if level == 0:
                    out_channels = img_channels
                else:
                    out_channels = model_channels * channel_mult[level - 1]
                    
                layers.append(nnx.Conv2d(channels, out_channels, kernel_size=3, padding=1))
                self.up_blocks.append(nnx.Sequential(*layers))
        
        # Output
        self.norm_out = nnx.GroupNorm(num_groups=32, num_features=model_channels)
        self.conv_out = nnx.Conv2d(model_channels, img_channels, kernel_size=3, padding=1)
        
    def __call__(self, x: jnp.ndarray, sigma: jnp.ndarray, labels: Optional[jnp.ndarray] = None,
                 training: bool = True) -> jnp.ndarray:
        # Embed noise level
        timestep_embedding = get_timestep_embedding(sigma)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        hs = [h]
        for block in self.down_blocks:
            h = block(h, training=training)
            hs.append(h)
        
        # Middle
        for block in self.middle_blocks:
            h = block(h, training=training)
        
        # Upsampling
        for block in self.up_blocks:
            if len(hs) > 0:
                h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = block(h, training=training)
        
        # Output
        h = self.norm_out(h)
        h = jax.nn.silu(h)
        h = self.conv_out(h)
        
        return h

def get_timestep_embedding(timesteps: jnp.ndarray, dim: int = 128, max_period: int = 10000) -> jnp.ndarray:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding 