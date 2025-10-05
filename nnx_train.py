import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

import os
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh

from models.nnx_precond import EDMPrecond
from absl import logging

from training.loss import edm_loss_fn
from clu import metric_writers, periodic_actions
from flax.training import common_utils
from configs import default

import input_pipeline
from jax.sharding import PartitionSpec as P, NamedSharding
from models.nnx_precond import EDMPrecondConfig
from utils.parallelism import TrainState
import utils.parallelism as parallelism_utils


def train_step(state: TrainState, batch: jax.Array, learning_rate_fn):
    rngs = state.rngs.fold_in(state.step)

    def loss_fn(params):
        module = nnx.merge(state.graphdef, params)
        module.set_attributes(deterministic=False)
        loss = edm_loss_fn(module, batch, rngs=rngs)
        return loss

    step = state.step
    lr = learning_rate_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    # metrics
    metrics = {"loss": loss, "learning_rate": lr}

    return new_state, metrics


def train_and_evaluate(config: default.Config, workdir: str):
    # Load Dataset
    logging.info("Loading dataset")
    dataset = input_pipeline.navier_stokes_data(config=config)
    train_iter = iter(dataset)

    # Build Model and Optimizer
    logging.info("Initializing model and optimizer")
    model_config = EDMPrecondConfig(
        model_type="DhariwalUNet",
        img_resolution=128,
        img_channels=1,
        label_dim=0,
        model_channels=128,
        channel_mult=[1, 1, 1, 2, 2],
        attn_resolutions=[16],
        num_blocks=1,
        dropout=0.0,
    )

    # Mesh definition
    devices_array = parallelism_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.parallelism.mesh_axes)

    # Data Sharding Specification
    data_sharding = NamedSharding(mesh, P(config.parallelism.data_sharding))

    start_step = 0
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    rng, inference_rng = jax.random.split(rng)

    def constructor(config: EDMPrecondConfig, key: jax.Array):
        return EDMPrecond(**vars(config), rngs=nnx.Rngs(params=key))

    # optimizer
    # slighlty different from original
    warmup_steps = config.warmup_steps
    base_lr = config.base_lr
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(base_lr),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = optax.adam(learning_rate=schedule)

    state, state_sharding = parallelism_utils.setup_initial_state(
        constructor, optimizer, model_config, init_rng, mesh
    )

    # test the sharding
    print(f"state_sharding: \n {state_sharding}")
    print(f"data sharding: \n {data_sharding}")
    print(f"mesh: \n {mesh}")

    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() == 0
    )

    jit_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_sharding,
            data_sharding,
            None,
        ),
        out_shardings=(state_sharding, None),
        static_argnames=("learning_rate_fn"),
        donate_argnums=0,
    )

    # Main Train Loops

    logging.info("Starting training loop")
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(num_profile_steps=5, logdir=workdir),
        ]

    train_metrics = []
    with metric_writers.ensure_flushes(writer):
        for step in range(start_step, config.num_train_steps):
            is_last_step = step == config.num_train_steps - 1

            # Shard data to devices and do a training step
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                batch = next(train_iter)
                # Single machine multi-GPU setup
                batch = jax.util.tree_map(
                    lambda x: jax.device_put(x, data_sharding), batch
                )
                # Multi-machine setup global_batch = jax.make_array_from_process_local_data(data_sharding, batch)
                state, metrics = jit_train_step(state, batch, schedule)
                train_metrics.append(metrics)

            if step & config.eval_every_steps == 0 or is_last_step:
                with report_progress.timed("training metrics"):
                    logging.info("Gather training metrics.")
                    train_metrics = common_utils.stack_forest(train_metrics)
                    # Compute different summary metrics here
                    lr = train_metrics.pop("learning_rate").mean()
                    metrics_sum = jax.util.tree_map(jnp.sum, train_metrics)
                    summary["learning_rate"] = lr

                    # Print metrics
                    summary = {"train_" + k: v for k, v in summary.items()}
                    writer.write_scalars(step, summary)
                    train_metrics = []

    print("Training complete!")


# def main():
#     # Set up experiment directory
#     config = OmegaConf.load("configs/pretrain/nnx-navier-stokes.yaml")
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     exp_dir = os.path.join(
#         config.training.exp_dir, f"{config.training.exp_name}_{timestamp}"
#     )
#     os.makedirs(exp_dir, exist_ok=True)
#     ckpt_dir = os.path.join(exp_dir, "ckpt")
#     os.makedirs(ckpt_dir, exist_ok=True)

#     # Initialize logging systems
#     # loss_tracker = LossTracker(exp_dir, save_freq=config.training.save_freq)

#     # Set up devices and sharding
#     num_devices = jax.local_device_count()
#     mesh = jax.sharding.Mesh(
#         mesh_utils.create_device_mesh((1, num_devices)), ("data", "model")
#     )
#     model_sharding_spec = jax.sharding.PartitionSpec(None, "model")
#     model_sharding = jax.NamedSharding(mesh, model_sharding_spec)

#     data_sharding_spec = jax.sharding.PartitionSpec("data")
#     data_sharding = jax.NamedSharding(mesh, data_sharding_spec)

#     # Set up RNGs
#     key = jax.random.PRNGKey(42)
#     key_params, key_loss, key_dropout = jax.random.split(key, 3)
#     rngs = nnx.Rngs(
#         params=key_params,
#         loss=key_loss,
#         sampling=key_sampling,
#         loader=key_loader,
#         dropout=key_dropout,
#     )

#     # Batch size per device
#     # batch_size = config.training.batch_size
#     batch_size = 4
#     if batch_size % num_devices != 0:
#         batch_size = batch_size // num_devices * num_devices
#     assert batch_size % num_devices == 0, "Batch size must be divisible by num_devices"

#     # EDMPrecond for Navier-Stokes


#     # Create optimizer with warmup
#     warmup_steps = config.training.warmup_steps
#     lr_schedule = optax.warmup_cosine_decay_schedule(
#         init_value=config.training.lr_min,
#         peak_value=config.training.lr_max,
#         warmup_steps=warmup_steps,
#         decay_steps=config.training.num_steps,
#     )
#     optimizer = nnx.Optimizer(net, optax.adamw(lr_schedule))
#     # optimizer = nnx.Optimizer(net, optax.adamw(1e-4))

#     # Create scheduler
#     # scheduler = instantiate(config.scheduler)
#     scheduler = Scheduler(
#         num_steps=200, schedule="linear", timestep="poly-7", scaling="none"
#     )
#     sampler = DiffusionSampler(scheduler=scheduler)

#     # Create state
#     state = nnx.state((net, optimizer))
#     state = jax.device_put(state, (model_sharding, data_sharding))
#     nnx.update((net, optimizer), state)

#     print("model sharding")
#     jax.debug.visualize_array_sharding(net.model.map_layer0.weight.value)


#     @functools.partial(jax.pmap, axis_name="data")
#     def pmapped_train_step(model, optimizer, images):
#         loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
#         grads = jax.lax.pmean(grads, axis_name="data")
#         loss = jax.lax.pmean(loss, axis_name="data")

#         new_params = jax.tree_map(
#             lambda param, g: param - g * LEARNING_RATE, params, grads
#         )
#         return new_params, loss

#     ################################################################################
#     # https://docs.jax.dev/en/latest/distributed_data_loading.html
#     # Step 1: setup the Dataset for pure data parallelism (do once)
#     ################################################################################
#     ds = input_pipeline.navier_stokes_data(config)
#     ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())

#     ################################################################################
#     # Step 2: create a jax.Array of per-replica batches from the per-process batch
#     # produced from the Dataset (repeat every step). This can be used with batches
#     # produced by different data loaders as well!
#     ################################################################################
#     # Grab just the first batch from the Dataset for this example
#     per_process_batch = ds.as_numpy_iterator().next()

#     mesh = jax.make_mesh((jax.device_count(),), ("data",))
#     sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
#     global_batch_array = jax.make_array_from_process_local_data(
#         sharding, per_process_batch
#     )
#     ################################################################################
#     # Step 3: pmapped training step + full training loop

#     for epoch in range(num_epochs):
#         # Iterate over the entire dataset
#         for per_process_batch in ds.as_numpy_iterator():
#             # A. Create the global array for the current batch
#             global_batch_array = jax.make_array_from_process_local_data(
#                 sharding, per_process_batch
#             )

#             # B. Run the training step with the global array
#             # This function should be pmapped to run on all devices
#             loss, replicated_model, replicated_optimizer = pmapped_train_step(
#                 replicated_model, replicated_optimizer, global_batch_array
#             )

#         # Optional: Log or save model checkpoints here
#         print(f"Epoch {epoch} finished.")
#     ################################################################################

#     # dataloader, dataset = input_pipeline.get_datasets(
#     #     n_devices=jax.local_device_count(), config=config, mesh_axis_names=("data",)
#     # )

#     step = 0

#     print(f"Starting training from step {step}")
#     print(f"Training for {config.train.num_steps} steps")
#     print(f"Using {num_devices} devices with batch size {batch_size}")
#     print(f"Dataloader length: {len(dataloader)}")

#     for epoch in range(config.training.num_epochs):
#         for batch_idx, batch in enumerate(dataloader):
#             if step >= config.train.num_steps:
#                 break

#             if isinstance(batch, dict):
#                 images = jax.device_put(batch["target"], data_sharding)
#                 # labels= batch.get('label', None)
#             else:
#                 images = jax.device_put(batch, data_sharding)
#                 # labels = None

#             loss = train_step(rngs, net, optimizer, images)
#             # loss = train_step_temp(net, optimizer, images, rngs)

#             # Log loss
#             if jax.process_index() == 0:
#                 loss_tracker.update(step, loss)
#                 if config.log.wandb and step % config.log.wandb_freq == 0:
#                     wandb.log(
#                         {
#                             "loss": loss,
#                             "step": step,
#                         }
#                     )
#                 if step % config.log.print_freq == 0:
#                     print(f"Step={step}/{config.train.num_steps} Loss={loss:.6f}")
#                     loss_tracker.save_loss_plot()
#                     loss_tracker.save_loss_data()

#                     # Generate and save samples
#                     if step % config.log.sample_freq == 0 and step > 0:
#                         print(f"Generating samples at step {step}")
#                         x_start = sampler.get_start(
#                             ref_shape=(
#                                 batch_size,
#                                 images.shape[1],
#                                 images.shape[2],
#                                 images.shape[3],
#                             ),
#                             rngs=rngs,
#                         )
#                         sample = sampler.sample(model=net, x_start=x_start, rngs=rngs)
#                         sample = dataset.unnormalize(sample)
#                         sample_path = os.path.join(exp_dir, f"samples_step_{step}.png")
#                         save_samples(sample, sample_path, colorbar=True, cmap="RdYlBu")
#                 if step % config.log.save_freq == 0 and step > 0:
#                     checkpoints.save_checkpoint(ckpt_dir, state, step)
#             # Update the loss tracker with the current step
#             loss_tracker.update(step, loss)

#             if epoch == 0 and batch_idx == 0:
#                 print("data sharding")
#                 jax.debug.visualize_array_sharding(images[:, 0, 0, 0])

#             # Increment the global step counter (once per batch)
#             step += 1

#     # Final save of the loss plots
#     loss_tracker.save_loss_plot()
#     loss_tracker.save_loss_data()

#     # Generate final samples
#     print("Generating final samples")
#     x_start = sampler.get_start(
#         ref_shape=(batch_size, images.shape[1], images.shape[2], images.shape[3]),
#         rngs=rngs,
#     )
#     sample = sampler.sample(model=net, x_start=x_start, rngs=rngs)
#     sample = dataset.unnormalize(sample)
#     sample_path = os.path.join(exp_dir, "final_samples.png")
#     save_samples(sample, sample_path, colorbar=True, cmap="RdYlBu")
#     print(f"Saved final samples at {sample_path}.")

#     print("Training complete!")


if __name__ == "__main__":
    main()
