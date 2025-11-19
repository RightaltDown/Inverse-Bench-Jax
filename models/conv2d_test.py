import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
from typing import Sequence
import functools
import numpy as np

# --- Define dummy weight_init function for the nnx module ---
def weight_init(key, shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform": return np.sqrt(6 / (fan_in + fan_out)) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == "xavier_normal": return np.sqrt(2 / (fan_in + fan_out)) * jax.random.normal(key, shape)
    if mode == "kaiming_uniform": return np.sqrt(3 / fan_in) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == "kaiming_normal": return np.sqrt(1 / fan_in) * jax.random.normal(key, shape)

# --- Define the JAX/NNX Conv2d Class (Your implementation) ---
class Conv2d(nnx.Module):
    """2D convolution layer with optional up/downsampling."""

    def __init__( self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1, 1], fused_resample=False, init_mode="kaiming_normal", init_weight=1,init_bias=0,
        *, rngs: nnx.Rngs):
        assert not (up and down)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        
        weight_key, bias_key = jax.random.split(rngs.params())

        # NOTE: kaiming_normal maps to lecun_normal in standard JAX/Flax init
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel,fan_out=out_channels*kernel*kernel)
        self.weight = nnx.Param(weight_init(weight_key, [kernel, kernel, in_channels, out_channels], **init_kwargs) * init_weight)
        self.bias = nnx.Param(weight_init(bias_key, [out_channels], **init_kwargs) * init_bias) if bias else None

        if up or down:
            f = jnp.array(resample_filter, dtype=jnp.float32)
            f_outer = jnp.outer(f, f)
            f_reshaped = f_outer.reshape(f_outer.shape[0], f_outer.shape[1], 1, 1)
            self.resample_filter = nnx.Param(
                f_reshaped / (jnp.sum(f) ** 2), trainable=False
            )
        else:
            self.resample_filter = None

    def __call__(self, x):
        # Get parameters
        w = self.weight.value.astype(x.dtype) if self.weight is not None else None
        b = self.bias.value.astype(x.dtype) if self.bias is not None else None
        f = self.resample_filter.value.astype(x.dtype) if self.resample_filter is not None else None       
       
        # Correctly determine padding based on kernel size (odd kernel size like 3 gives w_pad=1)
        w_pad = w.shape[0] // 2 if w is not None else 0  # Using first dimension for HWIO format
        f_pad = (f.shape[0] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            f_up = jnp.tile(f * 4, (1, 1, 1, self.in_channels))  # HWIO format
            x = jax.lax.conv_general_dilated(
                x,
                f_up,
                window_strides=(1, 1),
                padding=[(max(f_pad - w_pad + 1, 1), max(f_pad - w_pad + 1, 1))] * 2,
                lhs_dilation=(2, 2),
                rhs_dilation=(1, 1),
                feature_group_count=self.in_channels,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            x = jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=[(max(w_pad - f_pad, 0), max(w_pad - f_pad, 0))] * 2,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

        elif self.fused_resample and self.down and w is not None:
            x = jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=[(w_pad + f_pad, w_pad + f_pad)] * 2,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            f_down = jnp.tile(f, (1, 1, 1, self.out_channels))  # [H, W, out_channels, 1]
            x = jax.lax.conv_general_dilated(
                x,
                f_down,
                window_strides=(2, 2),
                padding=[(0, 0)] * 2,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=self.out_channels,
            )

        else:
            if self.up:
                f_up = jnp.tile(f * 4, (1, 1, 1, self.in_channels))  # HWIO format
                x = jax.lax.conv_general_dilated(
                    x,
                    f_up,
                    window_strides=(1, 1),
                    padding=[(f_pad + 1, f_pad + 1)] * 2,
                    lhs_dilation=(2, 2),
                    rhs_dilation=(1, 1),
                    feature_group_count=self.in_channels,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )

            if self.down:
                f_down = jnp.tile(f, (1, 1, 1, self.in_channels))  # HWIO format
                x = jax.lax.conv_general_dilated(
                    x,
                    f_down,
                    window_strides=(2, 2),
                    padding=[(f_pad, f_pad)] * 2,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                    feature_group_count=self.in_channels,
                )

            if w is not None:
                x = jax.lax.conv_general_dilated(
                    x,
                    w,
                    window_strides=(1, 1),
                    padding=[(w_pad, w_pad)] * 2,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )

        # Add bias if needed
        if b is not None:
            x = jnp.add(x, b.reshape(1, 1, 1, -1))  # Reshape bias for NHWC format

        return x

# ----------------------------------------------------------------------
## Test Implementation
# ----------------------------------------------------------------------

# --- Test Parameters ---
batch_size = 2
img_size = 32
in_c = 3
out_c = 16
kernel_size = 3
key = jax.random.PRNGKey(42)

# Generate a single set of random data
data_key, init_key = jax.random.split(key)
dummy_input = jax.random.normal(data_key, (batch_size, img_size, img_size, in_c), dtype=jnp.float32)

# Use separate keys for each initialization
nnx_init_key, nnx_ref_key = jax.random.split(init_key)

# 1. Initialize Your Custom NNX Conv2d (Standard Case: No up/downsampling)
print("Testing Custom Conv2d (standard)...")
custom_conv = Conv2d(
    in_channels=in_c,
    out_channels=out_c,
    kernel=kernel_size,
    bias=True,
    init_weight=1,
    init_bias=0,
    rngs=nnx.Rngs(params=nnx_init_key),
)
custom_output = custom_conv(dummy_input)

# 2. Initialize NNX Reference Conv (using nnx.Conv)
print("Testing NNX nnx.Conv (reference)...")
nnx_ref_conv = nnx.Conv(
    in_features=in_c,
    out_features=out_c,
    kernel_size=(kernel_size, kernel_size),
    strides=(1, 1),
    padding='SAME',
    use_bias=True,
    rngs=nnx.Rngs(params=nnx_ref_key),
)

# Override reference params with custom params to ensure identical weights
# NNX Conv stores kernel as (kernel_h, kernel_w, in_features, out_features) = HWIO
# Your Conv2d already uses HWIO format, so direct assignment works
nnx_ref_conv.kernel.value = custom_conv.weight.value
nnx_ref_conv.bias.value = custom_conv.bias.value

# Run reference module with matched parameters
ref_output = nnx_ref_conv(dummy_input)

# 3. Compare Results
print("\nComparing outputs...")
print(f"Custom output shape: {custom_output.shape}")
print(f"Reference output shape: {ref_output.shape}")
print(f"Custom output mean: {jnp.mean(custom_output):.6f}")
print(f"Reference output mean: {jnp.mean(ref_output):.6f}")

try:
    jnp.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
    print("✅ Test Passed: Custom Conv2d output matches nnx.Conv output for the standard case!")
except AssertionError as e:
    print("❌ Test Failed: Outputs do not match.")
    print(f"Max absolute difference: {jnp.max(jnp.abs(custom_output - ref_output))}")
    print(e)


# ----------------------------------------------------------------------
## Test Case for Downsampling (Shape Check)
# ----------------------------------------------------------------------

print("\n" + "="*70)
print("Testing Custom Conv2d (Downsampling, Non-Fused) Shape...")
down_conv = Conv2d(
    in_channels=in_c,
    out_channels=out_c,
    kernel=kernel_size,
    bias=True,
    down=True,
    fused_resample=False,
    rngs=nnx.Rngs(params=jax.random.PRNGKey(10)),
)
down_output = down_conv(dummy_input)

# Expected output shape for downsampling
expected_h = img_size // 2
expected_w = img_size // 2

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {down_output.shape}")
print(f"Expected shape: ({batch_size}, {expected_h}, {expected_w}, {out_c})")

if down_output.shape == (batch_size, expected_h, expected_w, out_c):
    print(f"✅ Downsampling Shape Test Passed!")
else:
    print(f"❌ Downsampling Shape Test Failed!")

# ----------------------------------------------------------------------
## Test Case for Upsampling (Shape Check)
# ----------------------------------------------------------------------

print("\n" + "="*70)
print("Testing Custom Conv2d (Upsampling, Non-Fused) Shape...")
up_conv = Conv2d(
    in_channels=in_c,
    out_channels=out_c,
    kernel=kernel_size,
    bias=True,
    up=True,
    fused_resample=False,
    rngs=nnx.Rngs(params=jax.random.PRNGKey(20)),
)
up_output = up_conv(dummy_input)

# Expected output shape for upsampling
expected_h = img_size * 2
expected_w = img_size * 2

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {up_output.shape}")
print(f"Expected shape: ({batch_size}, {expected_h}, {expected_w}, {out_c})")

if up_output.shape == (batch_size, expected_h, expected_w, out_c):
    print(f"✅ Upsampling Shape Test Passed!")
else:
    print(f"❌ Upsampling Shape Test Failed!")