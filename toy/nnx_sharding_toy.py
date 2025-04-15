import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P


mesh = jax.sharding.Mesh(jax.local_devices(), ('model',))

class MLP(nnx.Module):
  def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
    self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

  def __call__(self, x):
    return self.linear2(jax.nn.relu(self.linear1(x)))

m = MLP(2, 64, 3, rngs=nnx.Rngs(0))
x = jnp.ones((32, 2))

model_spec = nnx.State(
  {
    'linear1': {'kernel': P(None, 'model')},
    'linear2': {'kernel': P('model', None)},
  }
)

@nnx.shard_map(
  mesh=mesh,
  in_specs=(nnx.StateSharding(model_spec), P(None)),
  out_specs=P(None),
)
def f(m, x):
  y = m(x)
  return jax.lax.psum(y, 'model')

y = f(m, x)
print('linear1')
jax.debug.visualize_array_sharding(m.linear1.kernel.value)
print('linear2')
jax.debug.visualize_array_sharding(m.linear2.kernel.value)