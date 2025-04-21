import os
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.4'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)



import torch
from utils.nnx_scheduler import Scheduler

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental import mesh_utils

import tensorflow as tf
import tensorflow_datasets as tfds
from flax import nnx

from models.nnx_precond import EDMPrecond

from utils.nnx_diffusion import DiffusionSampler

import matplotlib.pyplot as plt
from PIL import Image

def save_samples(samples, save_path, grid_size=None):
    """
    Save generated samples as a grid image.
    
    Args:
        samples: JAX array of shape [batch, height, width, channels]
        save_path: Path to save the output image
        grid_size: Tuple of (rows, cols) for the grid layout. If None, will be automatically determined.
    """
    # Convert from JAX array to numpy
    samples_np = np.array(samples)
    
    # Determine grid size if not provided
    batch_size = samples_np.shape[0]
    if grid_size is None:
        grid_size = (int(np.sqrt(batch_size)), int(np.ceil(batch_size / int(np.sqrt(batch_size)))))
    
    # Create the grid
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    
    # Flatten axes if needed to make indexing consistent
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    # Plot each sample
    for i in range(batch_size):
        if i < len(axes):
            # For grayscale images (channels=1)
            if samples_np.shape[-1] == 1:
                axes[i].imshow(samples_np[i, :, :, 0], cmap='gray')
            else:
                # For RGB images
                axes[i].imshow(samples_np[i])
            
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved samples to {save_path}")

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
    
def edm_loss_fn(rng, model, images, sigma_data=0.5, P_mean=-1.2, P_std=1.2, labels=None, augment_pipe=None):
    # Creating random noise
    rnd_normal = jax.random.normal(rng.loss(), (images.shape[0], 1, 1, 1))
    
    # Calculate sigma following log-normal distribution
    sigma = jnp.exp(rnd_normal * P_std + P_mean)
    
    # Calculate weight for loss function
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    
    # Apply augmentation if provided
    y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    
    # generate noise to add to image
    n = jax.random.normal(rng.loss(), y.shape) * sigma
    
    # Get denoised prediction from model
    D_yn = model(y + n, sigma, labels, augment_labels=augment_labels)
    
    # Calculate MSE loss with importance weighting
    loss = weight * ((D_yn - y) ** 2)
    return jnp.mean(loss)


def main():
    exp_dir = os.path.join("exps/pretrain", "EDM-MNIST")
    os.makedirs(exp_dir, exist_ok=True)
    
    # create a mesh + shardings
    rngs = nnx.Rngs(42, params=0, loss=1, sampling=2)
    num_devices = jax.local_device_count()
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((num_devices,)), ('data',)
    )
    model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
    
    batch_size = 128 // num_devices
    assert(
        batch_size * num_devices == 128
    ), "Batch size must be divisible by num processes"
    
    
    net = EDMPrecond(
        rngs=rngs,
        model_type='DhariwalUNet',
        img_resolution=32,
        img_channels=1,
        label_dim=0,
        model_channels=128,
        channel_mult=[1, 1, 1, 1],
        attn_resolutions= [16],
        num_blocks=1,
        dropout=0.0,
    )
    
    scheduler = Scheduler(num_steps=200, schedule='linear', timestep='poly-7', scaling='none')
    
    sampler = DiffusionSampler(scheduler=scheduler)
    
    optimizer = nnx.Optimizer(net, optax.adamw(1e-2))
    
    # replicate state
    state = nnx.state((net, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((net, optimizer), state)
    
    print('model sharding')
    jax.debug.visualize_array_sharding(net.model.map_layer0.weight.value)
    
    # nnx.display(net)
    # could potentially use nnx.shard_map here instead of jax.device_put on the images and labels
    # @partial(nnx.jit, static_argnums=(4,))
    @nnx.jit
    def train_step(model, optimizer: nnx.Optimizer, images, labels, rngs):
        def loss_fn(model):
            return edm_loss_fn(
                rngs,
                model,
                images,
                labels=None,
                augment_pipe=None   
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss    
    
    train_dataset = create_mnist_dataset(batch_size, split='train', shuffle=True)
    test_dataset = create_mnist_dataset(batch_size, split='test', shuffle=False)
    
    num_steps = 1e5
    num_epochs = (int) (256 * num_steps // len(train_dataset) + 1)
    
    print(f"Num epochs: {num_epochs}")
    
    training_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for (images, _) in train_dataset:
            if training_steps > 1000: break
            
            # shard data
            # images, labels = batch
            # images = jax.random.normal(rngs.loss(), (32, 32, 32, 1))
            images = jax.image.resize(images, (32, 32, 32, 1), 'linear')
            images = jax.device_put(images, data_sharding)
            
            
            # train
            loss = train_step(net, optimizer, images, labels=None, rngs=rngs)
            epoch_loss += loss
            batch_count += 1
            
            if training_steps % 100 == 0:
                print(f"Generating samples at step {training_steps}")
                
                x_start = sampler.get_start(ref_shape=(32, 32, 32, 1), rngs=rngs)
                sample = sampler.sample(model=net, x_start=x_start, rngs=rngs)                
                sample_path = os.path.join(exp_dir, f"samples_step_{training_steps}.png")
                
                save_samples(sample, sample_path)
            
            if epoch == 0 and batch_count == 1:
                print('data sharding')
                jax.debug.visualize_array_sharding(images[:,0,0,0])
        if epoch % 10 == 0:
            avg_loss = epoch_loss / batch_count
            print(f'epoch={epoch}, loss={avg_loss}')
        
        training_steps += 1
        
    
    final_state = nnx.state((net, optimizer))
    final_state = jax.device_get(final_state)
    
    print("Training complete!")
    # total = 0
    # correct = 0
    # for test_batch in test_dataset:
    #     test_images, test_labels = test_batch
    #     test_images = jax.device_put(test_images, model_sharding)
        
    #     # Forward pass
    #     logits = net(test_images)
    #     predictions = jnp.argmax(logits, axis=-1)
    #     true_labels = jnp.argmax(test_labels, axis=-1)
        
    #     # Compute accuracy
    #     correct += jnp.sum(predictions == true_labels)
    #     total += len(true_labels)
    
    # accuracy = correct / total
    # print(f'Test accuracy: {accuracy * 100:.2f}%')
    
    # dereplicate state
    state = nnx.state((net, optimizer))
    state = jax.device_get(state)
    nnx.update((net, optimizer), state)
    

if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], "GPU")
    main()
