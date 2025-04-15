import functools
import time
import os

from einops import repeat

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from typing import Any, Optional, Callable
from tqdm import tqdm, trange

import flax
from flax.training import train_state
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints
from flax import jax_utils
from flax import nnx

import optax

import jax.numpy as jnp
import numpy as np
import jax 

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

import wandb

import unet
import utils
from sampling import sample_loop, ddpm_sample_step, model_predict

def create_optimizer(config):

    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate = config.lr , b1=config.beta1, b2 = config.beta2, 
            eps=config.eps)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer

def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(dtype=model_dtype, **kwargs)

def initialized(key, image_size,image_channel, model):

  input_shape = (1, image_size, image_size, image_channel)

  @nnx.jit
  def init(*args):
    return model.init(*args)
  variables = init(
      {'params': key}, 
      jnp.ones(input_shape, model.dtype), # x noisy image
      jnp.ones(input_shape[:1], model.dtype) # t
      )

  return variables['params']


class TrainState(train_state.TrainState):
  params_ema: Any = None
  dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None

def create_train_state(rng, config: ml_collections.ConfigDict):
  """Creates initial `TrainState`."""

  dynamic_scale = None
  platform = jax.local_devices()[0].platform

  if config.training.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  model = create_model(
      model_cls=unet.Unet, 
      half_precision=config.training.half_precision,
      dim = config.model.dim, 
      out_dim =  config.data.channels,
      dim_mults = config.model.dim_mults)

  image_size = config.data.image_size
  input_dim = config.data.channels * 2 if config.ddpm.self_condition else config.data.channels
  params = initialized(rng, image_size, input_dim, model)

  tx = create_optimizer(config.optim)

  state = TrainState.create(
      apply_fn=model.apply, 
      params=params, 
      tx=tx, 
      params_ema=params,
      dynamic_scale=dynamic_scale)

  return state

def copy_params_to_ema(state):
   state = state.replace(params_ema = state.params)
   return state

def apply_ema_decay(state, ema_decay):
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1. - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema = params_ema)
    return state


def load_wandb_model(state, workdir, wandb_artifact):
    artifact = wandb.run.use_artifact(wandb_artifact, type='ddpm_model')
    artifact_dir = artifact.download(workdir)
    return checkpoints.restore_checkpoint(artifact_dir, state)


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def train(config: ml_collections.ConfigDict, 
    workdir: str,
    wandb_artifact: str = None) -> TrainState:
    """Execute model training loop.

    Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

    Returns:
    Final TrainState.
    """
    # create writer 
    writer = metric_writers.create_default_writer(
    logdir=workdir, just_logging=jax.process_index() != 0)
    # set up wandb run
    if config.wandb.log_train and jax.process_index() == 0:
        wandb_config = utils.to_wandb_config(config)
    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        job_type=config.wandb.job_type,
        config=wandb_config)
    # set default x-axis as 'train/step'
    #wandb.define_metric("*", step_metric="train/step")

    sample_dir = os.path.join(workdir, "samples")

    # rng = jax.random.PRNGKey(config.seed)
    rng = nnx.Rng(config.seed)

    # rng, d_rng = jax.random.split(rng) 
    train_iter = get_dataset(rng, config)

    num_steps = config.training.num_train_steps

    state = create_train_state(rng, config)
    # if wandb_artifact is not None:
    #     logging.info(f'loading model from wandb: {wandb_artifact}')
    #     state = load_wandb_model(state, workdir, wandb_artifact)
    # else:
    #     state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    loss_fn = get_loss_fn(config)

    ddpm_params = utils.get_ddpm_params(config.ddpm)
    ema_decay_fn = create_ema_decay_schedule(config.ema)
    train_step = functools.partial(p_loss, ddpm_params=ddpm_params, loss_fn =loss_fn, self_condition=config.ddpm.self_condition, is_pred_x0=config.ddpm.pred_x0, pmap_axis ='batch')
    p_train_step = jax.pmap(train_step, axis_name = 'batch')
    p_apply_ema = jax.pmap(apply_ema_decay, in_axes=(0, None), axis_name = 'batch')
    p_copy_params_to_ema = jax.pmap(copy_params_to_ema, axis_name='batch')

    train_metrics = []
    hooks = []

    sample_step = functools.partial(ddpm_sample_step, ddpm_params=ddpm_params, self_condition=config.ddpm.self_condition, is_pred_x0=config.ddpm.pred_x0)
    p_sample_step = jax.pmap(sample_step, axis_name='batch')

    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    for step, batch in zip(tqdm(range(step_offset, num_steps)), train_iter):
        state, metrics = p_train_step(rng, state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')
            logging.info(f"Number of devices: {batch['image'].shape[0]}")
            logging.info(f"Batch size per device {batch['image'].shape[1]}")
            logging.info(f"input shape: {batch['image'].shape[2:]}")

        # update state.params_ema
        if (step + 1) <= config.ema.update_after_step:
            state = p_copy_params_to_ema(state)
        elif (step + 1) % config.ema.update_every == 0:
            ema_decay = ema_decay_fn(step)
            logging.info(f'update ema parameters with decay rate {ema_decay}')
            state =  p_apply_ema(state, ema_decay)

        if config.training.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.training.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train/{k}': v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary['time/seconds_per_step'] =  (time.time() - train_metrics_last_t) /config.training.log_every_steps

                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

                if config.wandb.log_train:
                    wandb.log({
                        "train/step": step, ** summary
                    })

        # Save a checkpoint periodically and generate samples.
        if (step + 1) % config.training.save_and_sample_every == 0 or step + 1 == num_steps:
            # generate and save sampling 
            logging.info(f'generating samples....')
            samples = []
            for i in trange(0, config.training.num_sample, config.data.batch_size):
                rng, sample_rng = jax.random.split(rng)
                samples.append(sample_loop(sample_rng, state, tuple(batch['image'].shape), p_sample_step, config.ddpm.timesteps))
            samples = jnp.concatenate(samples) # num_devices, batch, H, W, C
            
            this_sample_dir = os.path.join(sample_dir, f"iter_{step}_host_{jax.process_index()}")
            tf.io.gfile.makedirs(this_sample_dir)
            
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            samples_array = utils.save_image(samples, config.training.num_sample, fout, padding=2)
            if config.wandb.log_sample:
                utils.wandb_log_image(samples_array, step+1)
            # save the chceckpoint
            save_checkpoint(state, workdir)
            if step + 1 == num_steps and config.wandb.log_model:
                utils.wandb_log_model(workdir, step+1)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return(state)