import os
os.environ['HYDRA_FULL_ERROR'] = '1'
from omegaconf import OmegaConf
import hydra

import torch
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental import mesh_utils
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

from flax import nnx
from flax.training import train_state
import functools
from jax.sharding import PartitionSpec as P

# JAX data loading functions
def create_mnist_dataset(batch_size, split='train', shuffle=True):
    ds = tfds.load('mnist', split=split, as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, 10)     # One-hot encode for classification
        return image, label
    
    ds = ds.map(preprocess)
    
    if shuffle:
        ds = ds.shuffle(10000)
    
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(ds)

def create_pmap_datasets(batch_size, num_devices):
    per_device_batch = batch_size // num_devices
    total_batch = per_device_batch * num_devices
    
    train_dataset = create_mnist_dataset(total_batch, split='train', shuffle=True)
    test_dataset = create_mnist_dataset(total_batch, split='test', shuffle=False)
    
    return train_dataset, test_dataset

class TrainState(train_state.TrainState):
    counts: nnx.State
    graphdef: nnx.GraphDef

class Count(nnx.Variable[nnx.A]):
  pass


# Validation step for model evaluation
def eval_step(state: TrainState, batch):
    """Parallel validation step"""
    model = nnx.merge(state.graphdef, state.params, state.counts)
    _, aux = state.loss_fn(model, batch, is_training=False)
    
    return state, aux

# def train_step(state: TrainState, batch, config=None):
#     """parallel train step that uses the loss_fn stored in state"""
#     def mnist_loss_fn(params):
#         model = nnx.merge(state.graphdef, params, state.counts)
#         logits = model(batch['image'])
#         loss = optax.softmax_cross_entropy_with_integer_labels(
#             logits=logits, labels=batch['label']
#         ).mean()
        
#         # Return additional metrics in aux dict
#         aux = {
#             'accuracy': (logits.argmax(axis=-1) == batch['label']).mean(),
#             'prediction_error': jnp.abs(logits - batch['label']).mean(),
#             'logits': logits,
#         }
#         loss = jax.lax.pmean(loss, axis_name='data')
#         return loss, aux

#     (loss, aux), grads = jax.value_and_grad(mnist_loss_fn, has_aux=True)(state.params)
#     # (loss, (aux, model)), grads = jax.value_and_grad(mnist_loss_fn, has_aux=True)(state.params)
#     # loss, grads = nnx.value_and_grad(loss_fn)(model) # jax example implementation
#     # grads = jax.lax.pmean(grads, axis_name='data')
    
#     state = state.apply_gradients(grads=grads)

#     return state, {"loss": loss, **aux}


@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="mnist")
def main(config):
    
    exp_dir = os.path.join(config.log.exp_dir, config.log.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # create a mesh + shardings
    num_devices = jax.local_device_count()
    # mesh = jax.sharding.Mesh(
    #     mesh_utils.create_device_mesh((num_devices,)), ('data',)
    # )
    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))
    model_sharding = jax.NamedSharding(mesh, P())
    data_sharding = jax.NamedSharding(mesh, P('data'))
    
    batch_size = config.train.batch_size // num_devices
    assert(
        batch_size * num_devices == config.train.batch_size
    ), "Batch size must be divisible by num processes"
    
    class CNN(nnx.Module):
        """A simple CNN model."""

        def __init__(self, 
                 in_channels=1,         # Default: MNIST grayscale images
                 hidden_dim1=32,        # First conv layer channels 
                 hidden_dim2=64,        # Second conv layer channels
                 hidden_dim3=256,       # First linear layer size
                 num_classes=10,        # Default: MNIST has 10 classes (digits 0-9)
                 kernel_size=(3, 3),    # Kernel size for conv layers
                 pool_size=(2, 2),      # Pool size for avg_pool
                 pool_stride=(2, 2),    # Pool stride
                 flatten_size=3136,     # Size after flattening (depends on input image size)
                 *,
                 rngs: nnx.Rngs):
        
            self.conv1 = nnx.Conv(in_channels, hidden_dim1, kernel_size=kernel_size, rngs=rngs)
            self.conv2 = nnx.Conv(hidden_dim1, hidden_dim2, kernel_size=kernel_size, rngs=rngs)
            self.avg_pool = partial(nnx.avg_pool, window_shape=pool_size, strides=pool_stride)
            self.linear1 = nnx.Linear(flatten_size, hidden_dim3, rngs=rngs)
            self.linear2 = nnx.Linear(hidden_dim3, num_classes, rngs=rngs)

        def __call__(self, x):
            x = self.avg_pool(nnx.relu(self.conv1(x)))
            x = self.avg_pool(nnx.relu(self.conv2(x)))
            x = x.reshape(x.shape[0], -1)  # flatten
            x = nnx.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = CNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-2))
    
    # replicate state
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((model, optimizer), state)
    
    print('model sharding')
    jax.debug.visualize_array_sharding(model.linear1.kernel.value)
    
    
    # @nnx.shard_map(
    #     mesh=mesh, in_specs=(P(None), P('data')), out_specs=P('data')
    # )
    @nnx.jit
    def train_step(model: CNN, optimizer: nnx.Optimizer, x, y):
        def loss_fn(model: CNN):
            y_pred = model(x)
            loss = optax.softmax_cross_entropy(
                logits=y_pred,
                labels=y
            ).mean()
            
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss    
    
    train_dataset = create_mnist_dataset(batch_size, split='train', shuffle=True)
    test_dataset = create_mnist_dataset(batch_size, split='test', shuffle=False)
    num_epochs = 20
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in  train_dataset:
            # shard data
            images, labels = batch
            images, labels = jax.device_put((images, labels), data_sharding)
            
            # train
            loss = train_step(model, optimizer, images, labels)
            epoch_loss += loss
            batch_count += 1
            
            if epoch == 0 and batch_count == 1:
                print('data sharding')
                jax.debug.visualize_array_sharding(images[:,0,0,0])
        if epoch % 10 == 0:
            avg_loss = epoch_loss / batch_count
            print(f'epoch={epoch}, loss={avg_loss}')
    
    total = 0
    correct = 0
    for test_batch in test_dataset:
        test_images, test_labels = test_batch
        test_images = jax.device_put(test_images, model_sharding)
        
        # Forward pass
        logits = model(test_images)
        predictions = jnp.argmax(logits, axis=-1)
        true_labels = jnp.argmax(test_labels, axis=-1)
        
        # Compute accuracy
        correct += jnp.sum(predictions == true_labels)
        total += len(true_labels)
    
    accuracy = correct / total
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    
    # dereplicate state
    state = nnx.state((model, optimizer))
    state = jax.device_get(state)
    nnx.update((model, optimizer), state)
    

if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
