import jax.numpy as jnp
from flax import nnx


# Example function to test broadcasting
@nnx.jit
def broadcasted_mul(x, beta):
    # x: [batch, height, width, channels]
    # beta: [batch] or [batch, 1, 1, 1]
    return x * beta


def main():
    batch_size = 4
    height = width = 32
    channels = 1
    # Create input data
    x = jnp.ones((batch_size, height, width, channels)) * jnp.arange(
        1, batch_size + 1
    ).reshape(-1, 1, 1, 1)
    beta = jnp.array([0.1, 0.2, 0.3, 0.4])  # shape [batch]
    # Test implicit broadcasting
    result_jit = broadcasted_mul(x, beta.reshape(-1, 1, 1, 1))
    result_jax = x * beta.reshape(-1, 1, 1, 1)
    print("Result from nnx.jit:", result_jit.shape)
    print("Result from JAX:", result_jax.shape)
    print("Are they equal?", jnp.allclose(result_jit, result_jax))
    # Test with beta as shape [batch] (should broadcast)
    result_jit2 = broadcasted_mul(x, beta)
    result_jax2 = x * beta.reshape(-1, 1, 1, 1)
    print("Result from nnx.jit (beta [batch]):", result_jit2.shape)
    print("Are they equal?", jnp.allclose(result_jit2, result_jax2))


if __name__ == "__main__":
    main()
