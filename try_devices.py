import os
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Use only GPU 3

# Don't force specific platform settings that might cause issues
# Let JAX use the available GPU
os.environ.pop('JAX_PLATFORMS', None)
os.environ.pop('XLA_FLAGS', None)

print("=== JAX Configuration ===")
for key in sorted([k for k in os.environ.keys() if 'JAX' in k or 'XLA' in k or 'CUDA' in k]):
    print(f"{key}: {os.environ[key]}")
print("========================")

import jax
print(f"Number of devices: {jax.device_count()}")
print(f"Local device count: {jax.local_device_count()}")
print(f"Devices: {jax.devices()}")

# Test standard GPU usage without virtual devices
import jax.numpy as jnp
from jax.experimental import mesh_utils

print(f"\nJAX version: {jax.__version__}")

# Simulate multiple devices using data parallelism
batch_size = 128
num_shards = 4  # We want 4 virtual devices
shard_size = batch_size // num_shards

print(f"\nSimulating {num_shards} devices with a batch size of {batch_size}")
print(f"Each shard will process {shard_size} examples")

# Create a simple loop that processes data in shards
for i in range(num_shards):
    print(f"Processing shard {i+1}/{num_shards} on the GPU") 