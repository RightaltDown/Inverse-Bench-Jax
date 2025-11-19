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
    rngs = jax.random.fold_in(state.rng, state.step)

    def loss_fn(params):
        module = nnx.merge(state.graphdef, params)
        # module.set_attributes(deterministic=False)
        # module.set_attributes(deterministic=False, decode=False)
        # Extract the target images from the batch dictionary
        images = batch['target']
        loss = edm_loss_fn(module, images, rngs=rngs)
        return loss

    step = state.step
    lr = learning_rate_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    # metrics
    metrics = {"loss": loss, "learning_rate": lr}
    
    return new_state, metrics


def train_and_evaluate(config: default.Config, workdir: str):
    # Load Dataset
    logging.info("Loading dataset")
    dataset, test_dataset = input_pipeline.navier_stokes_data(config=config)
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
        dropout=0.0
    )

    # Mesh definition
    devices_array = parallelism_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.parallelism.mesh_axes)

    # Data Sharding Specification
    data_sharding = NamedSharding(mesh, P(config.parallelism.data_sharding))

    start_step = 0
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, inference_rng, state_rng = jax.random.split(rng, 4)
    rngs = nnx.Rngs(params=init_rng, dropout=inference_rng, state=state_rng)

    def constructor(input_config: EDMPrecondConfig, rngs: nnx.Rngs):
        return EDMPrecond(**vars(input_config), rngs=rngs)

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
        constructor, optimizer, model_config, rngs, mesh
    )
    
    # visualize the sharding of the model
    # jax.debug.visualize_array_sharding(state.params.model.map_layer0.weight.value)

    # test the sharding
    # print(f"state_sharding: \n {state_sharding}")
    print(f"data sharding: \n {data_sharding}")
    # print(f"mesh: \n {mesh}")
    
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() == 0
    )
    

    jit_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_sharding,
            data_sharding,
        ), # type: ignore
        out_shardings=(state_sharding, None), # type: ignore
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
                # Single machine multi-GPU setup
                
                batch = jax.device_put(next(train_iter), data_sharding) # Multi-machine setup global_batch = jax.make_array_from_process_local_data(data_sharding, batch)
                if step == 0: jax.debug.visualize_array_sharding(batch['target'][:, 0, 0, 0])
                state, metrics = jit_train_step(state, batch, schedule)
                train_metrics.append(metrics)

            if step & config.eval_every_steps == 0 or is_last_step:
                with report_progress.timed("training metrics"):
                    logging.info("Gather training metrics.")
                    metrics_stacked = common_utils.stack_forest(train_metrics)
                    summary = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_stacked)
                    writer.write_scalars(step, {"train_" + k: v for k, v in summary.items()})
                    train_metrics = []

    print("Training complete!")