import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
x = jnp.ones((32, 2))

@nnx.shard_map(
  mesh=mesh, in_specs=(P(None), P('data')), out_specs=P('data')
)
def f(m, x):
  return m(x)

y = f(m, x)

jax.debug.visualize_array_sharding(y)