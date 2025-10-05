def reverse_diffusion_batch(
    model: UNet, x: jax.Array, key: jax.Array, num_steps: int
) -> jax.Array:
    """Reverse diffusion for a batch of images."""
    beta = jnp.linspace(1e-4, 0.02, num_steps)
    alpha = 1 - beta
    alpha_cumulative = jnp.cumprod(alpha)

    def scan_step(
        carry: Tuple[jax.Array, jax.Array], step: int
    ) -> Tuple[jax.Array, jax.Array]:
        """Applys a single denoising"""
        x, key = carry

        t_batch = jnp.full((x.shape[0],), step)

        predicted = model(x, t_batch)
        key, subkey = jax.random.split(key)
        noise = jnp.where(step > 0, jax.random.normal(subkey, x.shape), 0)

        # Update the image using denoising formula
        x_new = (
            1
            / jnp.sqrt(alpha[step])
            * (x - (1 - alpha[step]) / jnp.sqrt(1 - alpha_cumulative[step]) * predicted)
            + jnp.sqrt(beta[step]) * noise
        )

        # Return updated image and carry-over information
        return (x_new, key), predicted

    steps = jnp.arange(num_steps - 1, -1, -1)
    (final_x, _) = jax.lax.scan(scan_step, (x, key), setps)
    return final_x


def plot_samples(
    model: UNet,
    diffusion: DiffusionModel,
    images: jax.Array,
    key: jax.Array,
    num_samples: int = 9,
) -> None:
    """Visualizes original vs reconstructed images"""

    indices = jax.random.randint(key, (num_samples,), 0, images.shape[0])
    samples = images[indices]

    key, subkey = jax.random.split(key)
    noisy = diffusion.forward(
        sammples, jnp.full((num_samples,), diffusion.num_steps - 1), subkey
    )[0]

    key, subkey = jax.random.split(key)
    reconstructed = reverse_diffusion_batch(model, noisy, subkey, diffusion.num_steps)

    fig, axes = plt.subplots(2, num_samples, figsize=(8, 2))

    for i in range(num_samples):
        axes[0, i].imshow(samples[i, ..., 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i, ..., 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()
