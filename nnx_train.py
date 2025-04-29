import os
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.3'
os.environ['NCCL_DEBUG']='INFO'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use only GPU 3

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


from utils.nnx_scheduler import Scheduler
import os
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
from utils.misc import save_samples

from torch.utils.data import DataLoader
from training.dataset import LMDBData
from utils.loss_tracker import LossTracker


def create_navier_stokes_data(batch_size):
    root = "data/navier-stokes-train/Re200.0-t5.0"
    num_train_workers = 0

    dataset = LMDBData(root=root)
    print(dataset.length)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_train_workers,
            pin_memory=True,
            drop_last=True,
        )
    return dataloader 

def create_mnist_dataset(batch_size, split='train', shuffle=True):
    ds = tfds.load('mnist', split=split, as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0 
        image = image * 2.0 - 1.0
        return image
    
    ds = ds.map(preprocess)
    
    if shuffle:
        ds = ds.shuffle(10000)
    
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(ds)
    
def edm_loss_fn(rngs, model, images, sigma_data=0.5, P_mean=-1.2, P_std=1.2, labels=None, augment_pipe=None):
    rng_sigma, rng_noise = jax.random.split(rngs.loss())
    rnd_normal = jax.random.normal(rng_sigma, (images.shape[0], 1, 1, 1))
    sigma = jnp.exp(rnd_normal * P_std + P_mean)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    
    # Apply augmentation if provided
    y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    
    # Generate noise to add to image
    noise = jax.random.normal(rng_noise, y.shape)
    noised_input = y + noise * jnp.reshape(sigma, (-1, 1, 1, 1))
    
    # Get denoised prediction from model
    D_yn = model(noised_input, sigma, labels, augment_labels=augment_labels)
    
    # Calculate MSE loss with importance weighting
    mse = ((D_yn - y) ** 2)
    loss = weight * jnp.mean(mse, axis=(1, 2, 3))  # Average over spatial dimensions first
    return jnp.mean(loss)  # Then average over batch

def process_data(images, batch_size, dataset_type='MNIST'):
    if dataset_type == 'MNIST':
        images = jax.numpy.array(images)
        images = jax.image.resize(images, (batch_size, 32, 32, 1), 'linear')
        # Clip to ensure we stay in [-1, 1] after resize
        images = jnp.clip(images, -1.0, 1.0)
    elif dataset_type == 'Navier-Stokes':
        images = images['target'].detach().cpu().numpy()
        images = jax.numpy.array(images.reshape(batch_size, images.shape[2], images.shape[3], images.shape[1]))
        # # Scale Navier-Stokes data to [-1, 1]
        # min_val = jnp.min(images)
        # max_val = jnp.max(images)
        # images = 2 * (images - min_val) / (max_val - min_val) - 1
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return images

def main():
    # exp_dir = os.path.join("exps/pretrain", "LMBDtest")
    exp_dir = os.path.join("exps/pretrain", "MNIST")
    # exp_dir = os.path.join("exps/pretrain", "Navier-Stokes")
    os.makedirs(exp_dir, exist_ok=True)
    loss_tracker = LossTracker(exp_dir, save_freq=100)
    num_devices = jax.local_device_count()
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((num_devices,)), ('data',)
    )
    model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
    
    # Set up RNGs
    rngs = nnx.Rngs(42, params=0, loss=1, sampling=2, loader=3)
    
    # Batch size per device
    batch_size = 64
    if batch_size % num_devices != 0:
        batch_size = batch_size // num_devices * num_devices
    
    assert batch_size % num_devices == 0 

    
    # EDMPrecond for MNIST - adjust model parameters for [0,1] range
    net = EDMPrecond(
        rngs=rngs,
        model_type='DhariwalUNet',
        img_resolution=32,
        img_channels=1,
        label_dim=0,
        model_channels=32,
        channel_mult=[1, 1, 1, 1],
        attn_resolutions=[16],
        num_blocks=1,
        dropout=0.0,
    )
    
    # EDMPrecond for Navier-Stokes
    # net = EDMPrecond(
    #     rngs=rngs,
    #     model_type='DhariwalUNet',
    #     img_resolution=128,
    #     img_channels=1,
    #     label_dim=0,
    #     model_channels=128,
    #     channel_mult=[1, 1, 1, 2, 2],
    #     attn_resolutions= [16],
    #     num_blocks=1,
    #     dropout=0.0,
    # )
    
    scheduler = Scheduler(num_steps=200, schedule='linear', timestep='poly-7', scaling='none')
    sampler = DiffusionSampler(scheduler=scheduler)
    optimizer = nnx.Optimizer(net, optax.adamw(1e-2))
    
    # replicate state
    state = nnx.state((net, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((net, optimizer), state)
    
    print('model sharding')
    jax.debug.visualize_array_sharding(net.model.map_layer0.weight.value)
    
    @nnx.jit
    def train_step(model, optimizer: nnx.Optimizer, images, labels, rngs):
        def loss_fn(model):
            return edm_loss_fn(
                rngs,
                model,
                images,
                sigma_data=0.5,
                P_mean=-1.2,
                P_std=1.2,
                labels=None,
                augment_pipe=None   
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss    
    
    train_dataset = create_mnist_dataset(batch_size, split='train', shuffle=True)
    # train_dataset = create_navier_stokes_data(batch_size)
    
    print(len(train_dataset))
    print(f"Batch size per shard: {batch_size}")
    
    # Set training parameters
    num_steps = int(1e5)  # Total number of steps to train for
    print_every = 1_000
    steps_per_epoch = len(train_dataset) // batch_size
    num_epochs = (num_steps // steps_per_epoch) + 1
    
    print(f"Training for {num_steps} steps across approximately {num_epochs} epochs")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Step counter for global training progress
    step = 0
    
    # For tracking loss averages
    total_loss = 0.0
    total_count = 0
    epoch_loss = 0.0
    batch_count = 0
    
    for epoch in range(num_epochs):
        total_batch_loss = 0
        
        # training loop
        for batch_idx, images in enumerate(train_dataset):
            # Break out of the loop if we've reached the desired number of steps
            if step >= num_steps:
                break
                
            # Process data across device shards       
            processed_images = process_data(images, batch_size, dataset_type='MNIST')
            if step == 0: print(processed_images.shape)
            images = jax.device_put(processed_images, data_sharding)
            loss = train_step(net, optimizer, images, labels=None, rngs=rngs)
            
            # Add to the running totals
            total_loss += loss
            total_batch_loss += loss
            
            # Update the loss tracker with the current step
            loss_tracker.update(step, loss)
            
            if (step % print_every) == 0 or step == num_steps - 1:
                avg_loss = total_loss / total_count
                print(f"Step={step} Loss={avg_loss:.6f}")
                loss_tracker.update(step, total_loss, total_count)
                total_loss = 0
                total_count = 0            
            
            # Generate and save samples at regular intervals
            if step > 0 and (step % (print_every * 5) == 0 or step == num_steps - 1):
                print(f"Generating samples at step {step}")
                x_start = sampler.get_start(ref_shape=(batch_size, processed_images.shape[1], processed_images.shape[2], processed_images.shape[3]), rngs=rngs)
                sample = sampler.sample(model=net, x_start=x_start, rngs=rngs)                
                sample_path = os.path.join(exp_dir, f"samples_step_{step}.png")
                save_samples(sample, sample_path)
                print(f"Saved samples at {sample_path}.")
            
            if epoch ==0 and batch_idx == 0:
                print('data sharding')
                jax.debug.visualize_array_sharding(images[:,0,0,0])
            
            # Increment the global step counter (once per batch)
            step += 1
            total_count += 1
        epoch_loss += total_batch_loss
        batch_count += 1
        if batch_count > 0 and (epoch % 10) == 0:
            avg_epoch_loss = epoch_loss / steps_per_epoch
            print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.6f}")
            epoch_loss = 0.0
            batch_count = 0
    
    # Final save of the loss plots
    loss_tracker.save_loss_plot()
    loss_tracker.save_loss_data()
    
    # Generate final samples
    print("Generating final samples")
    x_start = sampler.get_start(ref_shape=(batch_size, 32, 32, 1), rngs=rngs)
    sample = sampler.sample(model=net, x_start=x_start, rngs=rngs)
    sample_path = os.path.join(exp_dir, "final_samples.png")
    save_samples(sample, sample_path)
    print(f"Saved final samples at {sample_path}.")
    
    print("Training complete!")

if __name__ == "__main__":
    main()