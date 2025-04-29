from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from models.nnx_modules import Linear, Conv2d, GroupNorm, PositionalEmbedding, FourierEmbedding, UNetBlock

# Functions from the second file
def weight_init(key, shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * jax.random.normal(key, shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * jax.random.normal(key, shape)
    raise ValueError(f'Invalid init mode "{mode}"')

# Import the module classes from the second file
from functools import partial 
import numpy as np
import jax
import jax.numpy as jnp

# Implementation of silu activation function (Swish)
def silu(x):
    return x * jax.nn.sigmoid(x)

# DhariwalUNet implementation using Flax nnx
class DhariwalUNet(nnx.Module):
    def __init__(self, rngs,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.
        
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4], # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32, 16, 8],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):        
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=jnp.sqrt(1/3), init_bias=jnp.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping network
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(rngs=rngs, in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(rngs=rngs, in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(rngs=rngs, in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(rngs=rngs, in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder
        self.enc = {}
        self.enc_names = []
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(
                    rngs=rngs, 
                    in_channels=cin, 
                    out_channels=cout, 
                    kernel=3, 
                    **init
                )
                self.enc_names.append(f'{res}x{res}_conv')
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(
                    rngs=rngs, 
                    in_channels=cout, 
                    out_channels=cout, 
                    down=True, 
                    **block_kwargs
                )
                self.enc_names.append(f'{res}x{res}_down')
            
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(
                    rngs=rngs, 
                    in_channels=cin, 
                    out_channels=cout, 
                    attention=(res in attn_resolutions), 
                    **block_kwargs
                )
                self.enc_names.append(f'{res}x{res}_block{idx}')
        
        # Calculate skip channel counts for decoder
        self.enc_modules = nnx.Dict()
        self.skip_channel_counts = []
        for key, module in self.enc.items():
            self.enc_modules[key] = module
            self.skip_channel_counts.append(module.out_channels)
        
        # Decoder
        self.dec = {}
        self.dec_names = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(
                    rngs=rngs, 
                    in_channels=cout, 
                    out_channels=cout, 
                    attention=True, 
                    **block_kwargs
                )
                self.dec_names.append(f'{res}x{res}_in0')
                self.dec[f'{res}x{res}_in1'] = UNetBlock(
                    rngs=rngs, 
                    in_channels=cout, 
                    out_channels=cout, 
                    **block_kwargs
                )
                self.dec_names.append(f'{res}x{res}_in1')
            else:        
                self.dec[f'{res}x{res}_up'] = UNetBlock(
                    rngs=rngs, 
                    in_channels=cout, 
                    out_channels=cout, 
                    up=True, 
                    **block_kwargs
                )
                self.dec_names.append(f'{res}x{res}_up')
            
            for idx in range(num_blocks + 1):
                # Get the skip connection channel count
                skip_channels = self.skip_channel_counts.pop()
                cin = cout + skip_channels
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(
                    rngs=rngs, 
                    in_channels=cin, 
                    out_channels=cout, 
                    attention=(res in attn_resolutions), 
                    **block_kwargs
                )
                self.dec_names.append(f'{res}x{res}_block{idx}')
        self.dec_modules = nnx.Dict()
        for key, module in self.dec.items():
            self.dec_modules[key] = module
            
        # Output layers
        self.out_norm = GroupNorm(rngs=rngs, num_channels=cout)
        self.out_conv = Conv2d(
            rngs=rngs, 
            in_channels=cout, 
            out_channels=out_channels, 
            kernel=3, 
            **init_zero
        )
        # print(f"Jax encoder: {[name for name in self.enc_names]}")
        # print(f"Jax decoder: {[name for name in self.dec_names]}")
        
        # assert 1 == 0

    def __call__(self, x, noise_labels, class_labels, augment_labels=None, train=True):
        # Mapping network
        emb = self.map_noise(noise_labels)
        
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
            
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        
        if self.map_label is not None:
            tmp = class_labels
            if train and self.label_dropout > 0:
                # JAX implementation of dropout for labels
                key = self.make_rng('dropout')
                mask = jax.random.bernoulli(key, 1 - self.label_dropout, (x.shape[0], 1))
                tmp = tmp * mask.astype(tmp.dtype)
            # here
            emb = emb + self.map_label(tmp)
        
        emb = silu(emb)

        # Encoder
        skips = []
        for name in self.enc_names:
            module = self.enc_modules.get(name)
            if isinstance(module, UNetBlock):
                x = module(x, emb, train=train)
            else:
                x = module(x)
            skips.append(x)
        
        # Decoder
        for name in self.dec_names:
            module = self.dec_modules.get(name)
            if x.shape[-1] != module.in_channels:  # Using -1 for channel dimension in NHWC
                skip = skips.pop()
                x = jnp.concatenate([x, skip], axis=-1)  # Concatenate along channel dimension
            x = module(x, emb, train=train)

        # Output
        x = silu(self.out_norm(x))
        x = self.out_conv(x)
        
        return x