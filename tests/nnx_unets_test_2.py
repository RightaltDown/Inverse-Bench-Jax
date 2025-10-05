import jax
import jax.numpy as jnp
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from flax import nnx

# Import the JAX/Flax implementation
from models.nnx_unets import DhariwalUNet
from models.nnx_precond import EDMPrecond
from models.unets import DhariwalUNet as torchUnet
from models.precond import EDMPrecond as torchEDMPrecond


def save_samples(
    samples,
    save_path,
    grid_size=None,
    colorbar=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
):
    """
    Save generated samples as a grid image with color bar.

    Args:
        samples: JAX or numpy array of shape [batch, height, width, channels] or [batch, channels, height, width]
        save_path: Path to save the output image
        grid_size: Tuple of (rows, cols) for the grid layout. If None, will be automatically determined.
        colorbar: Whether to add a colorbar to the figure
        cmap: Colormap to use for single-channel images
        vmin: Minimum value for color scaling (if None, uses data min)
        vmax: Maximum value for color scaling (if None, uses data max)
        title: Optional title for the figure
    """
    # Convert to numpy if not already
    if hasattr(samples, "detach"):  # PyTorch tensor
        samples_np = samples.detach().cpu().numpy()
    elif hasattr(samples, "block_until_ready"):  # JAX array
        samples_np = np.array(samples)
    else:
        samples_np = samples

    # Handle NCHW format (PyTorch) -> convert to NHWC
    if samples_np.shape[1] in [1, 3] and len(samples_np.shape) == 4:
        samples_np = samples_np.transpose(0, 2, 3, 1)

    # Determine grid size if not provided
    batch_size = samples_np.shape[0]
    if grid_size is None:
        grid_size = (
            int(np.sqrt(batch_size)),
            int(np.ceil(batch_size / int(np.sqrt(batch_size)))),
        )

    # Create the grid
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

    # Flatten axes if needed to make indexing consistent
    axes = np.array(axes).flatten() if rows > 1 or cols > 1 else np.array([axes])

    # Get global min/max for consistent colorbar if not provided
    if vmin is None:
        vmin = np.min(samples_np)
    if vmax is None:
        vmax = np.max(samples_np)

    # Plot each sample
    images = []
    for i in range(batch_size):
        if i < len(axes):
            # For grayscale images (channels=1)
            if samples_np.shape[-1] == 1:
                im = axes[i].imshow(
                    samples_np[i, :, :, 0], cmap=cmap, vmin=vmin, vmax=vmax
                )
                images.append(im)
            else:
                # For RGB images
                im = axes[i].imshow(samples_np[i])
                images.append(im)

            axes[i].axis("off")

    # Hide any unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    # Add a single colorbar if required and if we have single-channel images
    if colorbar and samples_np.shape[-1] == 1 and len(images) > 0:
        # Add a colorbar with some space for it
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label("Output Value")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def compare_tensors(
    torch_tensor, jax_tensor, rtol=1e-3, atol=1e-3, debug=False, name=""
):
    """
    Compare PyTorch and JAX tensors with given tolerances and provide detailed statistics

    Args:
        torch_tensor: PyTorch tensor
        jax_tensor: JAX array
        rtol: Relative tolerance
        atol: Absolute tolerance
        debug: Whether to print detailed debug info
        name: Name for this comparison (for logging)

    Returns:
        dict: Comparison statistics
        bool: Whether tensors match within tolerance
    """
    # Convert to numpy arrays
    torch_np = torch_tensor.detach().cpu().numpy()
    jax_np = np.array(jax_tensor)

    # Make sure they have the same shape
    if torch_np.shape != jax_np.shape:
        if debug:
            print(
                f"[{name}] Shape mismatch: JAX {jax_np.shape}, PyTorch {torch_np.shape}"
            )
        return {
            "match": False,
            "reason": "shape_mismatch",
            "jax_shape": jax_np.shape,
            "torch_shape": torch_np.shape,
        }, False

    # Calculate differences
    abs_diff = np.abs(torch_np - jax_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    median_diff = np.median(abs_diff)

    # Calculate relative differences where values are sufficiently large
    mask = np.abs(jax_np) > 1e-6
    rel_diff = np.zeros_like(abs_diff)
    if np.any(mask):
        rel_diff[mask] = abs_diff[mask] / np.abs(jax_np[mask])
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    # Calculate statistics
    torch_min, torch_max = np.min(torch_np), np.max(torch_np)
    jax_min, jax_max = np.min(jax_np), np.max(jax_np)

    # Create comparison statistics
    stats = {
        "match": np.allclose(torch_np, jax_np, rtol=rtol, atol=atol),
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "median_abs_diff": float(median_diff),
        "max_rel_diff": float(max_rel_diff),
        "mean_rel_diff": float(mean_rel_diff),
        "torch_range": (float(torch_min), float(torch_max)),
        "jax_range": (float(jax_min), float(jax_max)),
    }

    if debug:
        print(f"[{name}] Comparison results:")
        print(f"  Match within tolerance: {stats['match']}")
        print(f"  Max absolute difference: {stats['max_abs_diff']:.6e}")
        print(f"  Mean absolute difference: {stats['mean_abs_diff']:.6e}")
        print(f"  Max relative difference: {stats['max_rel_diff']:.6e}")
        print(
            f"  PyTorch range: [{stats['torch_range'][0]:.6f}, {stats['torch_range'][1]:.6f}]"
        )
        print(
            f"  JAX range: [{stats['jax_range'][0]:.6f}, {stats['jax_range'][1]:.6f}]"
        )

        # For smaller tensors, print the actual values
        if torch_np.size < 50:
            print(f"  PyTorch output:\n{torch_np}")
            print(f"  JAX output:\n{jax_np}")
            print(f"  Difference:\n{abs_diff}")

    return stats, stats["match"]


def jax_unet(
    x,
    noise_labels,
    class_labels,
    img_resolution,
    in_channels,
    out_channels,
    label_dim,
    model_channels,
    channel_mult,
    num_blocks,
    attn_resolutions,
    dropout,
    train=False,
    key=None,
):
    """
    Run forward pass through JAX UNet model
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    params_key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

    model_jax = DhariwalUNet(
        rngs=rngs,
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
    )

    # Forward pass in JAX
    output_jax = model_jax(x, noise_labels, class_labels, train=train)
    return output_jax, model_jax


def torch_unet(
    x,
    noise_labels,
    class_labels,
    img_resolution,
    in_channels,
    out_channels,
    label_dim,
    model_channels,
    channel_mult,
    num_blocks,
    attn_resolutions,
    dropout,
):
    """
    Run forward pass through PyTorch UNet model
    """
    model_torch = torchUnet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
    )

    output_torch = model_torch(x, noise_labels, class_labels)
    return output_torch, model_torch


def test_dhariwal_unet_comprehensive():
    """
    Test the Flax nnx implementation of DhariwalUNet against the PyTorch version
    with multiple configurations and edge cases.
    """
    print("\n===== COMPREHENSIVE DHARIWAL UNET TESTING =====")

    # Create output directory for visualizations
    os.makedirs("test_outputs/unet", exist_ok=True)

    # Track overall results
    all_results = {}
    all_passed = True

    # Test different configurations
    configs = [
        # Basic configuration
        {
            "name": "basic_config",
            "img_resolution": 32,
            "in_channels": 1,
            "out_channels": 1,
            "label_dim": 0,
            "model_channels": 64,
            "channel_mult": [1, 2, 3],
            "num_blocks": 1,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "batch_size": 4,
        },
        # Deeper network
        {
            "name": "deeper_network",
            "img_resolution": 64,
            "in_channels": 3,
            "out_channels": 3,
            "label_dim": 0,
            "model_channels": 64,
            "channel_mult": [1, 2, 2, 4],
            "num_blocks": 2,
            "attn_resolutions": [8, 16],
            "dropout": 0.1,
            "batch_size": 2,
        },
        # With class conditioning
        {
            "name": "class_conditioning",
            "img_resolution": 32,
            "in_channels": 1,
            "out_channels": 1,
            "label_dim": 10,
            "model_channels": 64,
            "channel_mult": [1, 2, 2],
            "num_blocks": 1,
            "attn_resolutions": [8],
            "dropout": 0.0,
            "batch_size": 4,
        },
        # High resolution
        {
            "name": "high_resolution",
            "img_resolution": 128,
            "in_channels": 1,
            "out_channels": 1,
            "label_dim": 0,
            "model_channels": 32,
            "channel_mult": [1, 1, 2, 2, 4],
            "num_blocks": 1,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "batch_size": 1,
        },
    ]

    for cfg in configs:
        print(f"\nTesting configuration: {cfg['name']}")

        # Extract configuration parameters
        img_resolution = cfg["img_resolution"]
        in_channels = cfg["in_channels"]
        out_channels = cfg["out_channels"]
        label_dim = cfg["label_dim"]
        model_channels = cfg["model_channels"]
        channel_mult = cfg["channel_mult"]
        num_blocks = cfg["num_blocks"]
        attn_resolutions = cfg["attn_resolutions"]
        dropout = cfg["dropout"]
        batch_size = cfg["batch_size"]

        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        key = jax.random.PRNGKey(42)

        # Create input data
        x = np.random.randn(batch_size, img_resolution, img_resolution, in_channels)
        x_jax = jnp.array(x)
        x_torch = torch.from_numpy(x).permute(0, 3, 1, 2)  # NHWC -> NCHW

        noise_labels = np.random.randn(batch_size)
        noise_labels_jax = jnp.array(noise_labels)
        noise_labels_torch = torch.from_numpy(noise_labels)

        # Handle class labels
        if label_dim > 0:
            class_indices = np.random.randint(0, label_dim, size=(batch_size,))
            class_labels_jax = jax.nn.one_hot(jnp.array(class_indices), label_dim)
            class_labels_torch = F.one_hot(
                torch.from_numpy(class_indices), label_dim
            ).float()
        else:
            class_labels_jax = None
            class_labels_torch = None

        # Test evaluation mode
        output_jax, _ = jax_unet(
            x_jax,
            noise_labels_jax,
            class_labels_jax,
            img_resolution,
            in_channels,
            out_channels,
            label_dim,
            model_channels,
            channel_mult,
            num_blocks,
            attn_resolutions,
            dropout,
            train=False,
            key=key,
        )

        output_torch, _ = torch_unet(
            x_torch,
            noise_labels_torch,
            class_labels_torch,
            img_resolution,
            in_channels,
            out_channels,
            label_dim,
            model_channels,
            channel_mult,
            num_blocks,
            attn_resolutions,
            dropout,
        )

        # Convert JAX output to match PyTorch format for comparison
        output_jax_nchw = output_jax.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # Compare outputs
        stats, passed = compare_tensors(
            output_torch,
            output_jax_nchw,
            rtol=1e-3,
            atol=1e-3,
            debug=True,
            name=f"UNet-{cfg['name']}",
        )

        # Save results
        all_results[f"UNet-{cfg['name']}"] = stats
        all_passed = all_passed and passed

        # Save sample visualizations
        save_samples(
            output_jax,  # Use NHWC format
            f"test_outputs/unet/{cfg['name']}_jax_output.png",
            grid_size=None,
            colorbar=True,
            title=f"JAX UNet Output ({cfg['name']})",
        )

        # Save PyTorch output (convert to NHWC first)
        output_torch_vis = output_torch.permute(0, 2, 3, 1).detach().cpu().numpy()
        save_samples(
            output_torch_vis,
            f"test_outputs/unet/{cfg['name']}_torch_output.png",
            grid_size=None,
            colorbar=True,
            title=f"PyTorch UNet Output ({cfg['name']})",
        )

        # Save difference visualization
        output_diff = np.abs(output_torch_vis - np.array(output_jax))
        save_samples(
            output_diff,
            f"test_outputs/unet/{cfg['name']}_difference.png",
            grid_size=None,
            colorbar=True,
            cmap="hot",
            title=f"Difference (Abs) ({cfg['name']})",
        )

        # Test with training mode
        if dropout > 0:
            print(f"Testing training mode with dropout={dropout}")

            # Use different keys for dropout during training
            key_train = jax.random.PRNGKey(43)
            torch.manual_seed(43)

            output_jax_train, _ = jax_unet(
                x_jax,
                noise_labels_jax,
                class_labels_jax,
                img_resolution,
                in_channels,
                out_channels,
                label_dim,
                model_channels,
                channel_mult,
                num_blocks,
                attn_resolutions,
                dropout,
                train=True,
                key=key_train,
            )

            # Check that outputs are different when in training mode due to dropout
            train_eval_diff = np.mean(
                np.abs(np.array(output_jax_train) - np.array(output_jax))
            )
            print(
                f"Mean difference between train and eval modes: {train_eval_diff:.6e}"
            )

            # We expect some difference due to dropout
            if train_eval_diff < 1e-6:
                print(
                    "WARNING: Train and eval modes produce identical outputs despite dropout"
                )
                all_passed = False

    print("\n===== DHARIWAL UNET TEST RESULTS =====")
    print(f"All tests passed: {all_passed}")

    return all_passed, all_results


def test_edm_precond_comprehensive():
    """
    Test the Flax nnx implementation of EDMPrecond against the PyTorch version
    with multiple configurations and edge cases.
    """
    print("\n===== COMPREHENSIVE EDM PRECOND TESTING =====")

    # Create output directory for visualizations
    os.makedirs("test_outputs/edm", exist_ok=True)

    # Track overall results
    all_results = {}
    all_passed = True

    # Test different configurations
    configs = [
        # Basic configuration
        {
            "name": "basic_config",
            "img_resolution": 32,
            "img_channels": 1,
            "label_dim": 0,
            "model_channels": 64,
            "channel_mult": [1, 2, 2],
            "num_blocks": 1,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "sigma_data": 0.5,
            "batch_size": 4,
            "sigma_min": 0.1,
            "sigma_max": 20.0,
        },
        # RGB images
        {
            "name": "rgb_images",
            "img_resolution": 64,
            "img_channels": 3,
            "label_dim": 0,
            "model_channels": 64,
            "channel_mult": [1, 2, 2, 4],
            "num_blocks": 2,
            "attn_resolutions": [8, 16],
            "dropout": 0.1,
            "sigma_data": 0.5,
            "batch_size": 2,
            "sigma_min": 0.02,
            "sigma_max": 80.0,
        },
        # With class conditioning
        {
            "name": "class_conditioning",
            "img_resolution": 32,
            "img_channels": 1,
            "label_dim": 10,
            "model_channels": 64,
            "channel_mult": [1, 2, 2],
            "num_blocks": 1,
            "attn_resolutions": [8],
            "dropout": 0.0,
            "sigma_data": 0.5,
            "batch_size": 4,
            "sigma_min": 0.1,
            "sigma_max": 20.0,
        },
        # Different sigma data
        {
            "name": "different_sigma_data",
            "img_resolution": 32,
            "img_channels": 1,
            "label_dim": 0,
            "model_channels": 64,
            "channel_mult": [1, 2, 2],
            "num_blocks": 1,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "sigma_data": 1.0,  # Different sigma_data
            "batch_size": 4,
            "sigma_min": 0.1,
            "sigma_max": 20.0,
        },
    ]

    for cfg in configs:
        print(f"\nTesting configuration: {cfg['name']}")

        # Extract configuration parameters
        img_resolution = cfg["img_resolution"]
        img_channels = cfg["img_channels"]
        label_dim = cfg["label_dim"]
        model_channels = cfg["model_channels"]
        channel_mult = cfg["channel_mult"]
        num_blocks = cfg["num_blocks"]
        attn_resolutions = cfg["attn_resolutions"]
        dropout = cfg["dropout"]
        sigma_data = cfg["sigma_data"]
        batch_size = cfg["batch_size"]
        sigma_min = cfg["sigma_min"]
        sigma_max = cfg["sigma_max"]

        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        key = jax.random.PRNGKey(42)

        # Create input data

        x = np.random.randn(batch_size, img_resolution, img_resolution, img_channels)
        x_jax = jnp.array(x)
        x_torch = torch.from_numpy(x).permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Create sigma values at different points in the diffusion process
        # Test edge cases and middle values
        sigmas_list = [
            np.full((batch_size,), sigma_min),  # Min sigma
            np.full((batch_size,), sigma_max),  # Max sigma
            np.full((batch_size,), np.sqrt(sigma_min * sigma_max)),  # Geometric mean
            np.random.uniform(sigma_min, sigma_max, (batch_size,)),  # Random values
        ]

        # Handle class labels
        if label_dim > 0:
            class_indices = np.random.randint(0, label_dim, size=(batch_size,))
            class_labels_jax = jax.nn.one_hot(jnp.array(class_indices), label_dim)
            class_labels_torch = F.one_hot(
                torch.from_numpy(class_indices), label_dim
            ).float()
        else:
            class_labels_jax = None
            class_labels_torch = None

        # Initialize models
        params_key, dropout_key = jax.random.split(key)
        rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

        model_jax = EDMPrecond(
            rngs=rngs,
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=label_dim,
            model_type="DhariwalUNet",
            model_channels=model_channels,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            num_blocks=num_blocks,
            dropout=dropout,
            sigma_data=sigma_data,
        )

        model_torch = torchEDMPrecond(
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=label_dim,
            model_type="DhariwalUNet",
            model_channels=model_channels,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            num_blocks=num_blocks,
            dropout=dropout,
            sigma_data=sigma_data,
        )

        # Test with different sigma values
        for i, sigmas in enumerate(sigmas_list):
            sigma_name = ["min", "max", "mid", "random"][i]
            sigma_jax = jnp.array(sigmas)
            sigma_torch = torch.from_numpy(sigmas)

            # Forward pass
            output_jax = model_jax(
                x_jax, sigma_jax, class_labels=class_labels_jax, train=False
            )
            output_torch = model_torch(
                x_torch, sigma_torch, class_labels=class_labels_torch
            )

            # Convert JAX output to match PyTorch format for comparison
            output_jax_nchw = output_jax.transpose(0, 3, 1, 2)  # NHWC -> NCHW

            # Compare outputs
            test_name = f"EDM-{cfg['name']}-sigma-{sigma_name}"
            stats, passed = compare_tensors(
                output_torch,
                output_jax_nchw,
                rtol=1e-3,
                atol=1e-3,
                debug=True,
                name=test_name,
            )

            # Save results
            all_results[test_name] = stats
            all_passed = all_passed and passed

            # Save sample visualizations for the random sigma case
            if sigma_name == "random":
                save_samples(
                    output_jax,  # Use NHWC format
                    f"test_outputs/edm/{cfg['name']}_jax_output.png",
                    grid_size=None,
                    colorbar=True,
                    title=f"JAX EDM Output ({cfg['name']})",
                )

                # Save PyTorch output (convert to NHWC first)
                output_torch_vis = (
                    output_torch.permute(0, 2, 3, 1).detach().cpu().numpy()
                )
                save_samples(
                    output_torch_vis,
                    f"test_outputs/edm/{cfg['name']}_torch_output.png",
                    grid_size=None,
                    colorbar=True,
                    title=f"PyTorch EDM Output ({cfg['name']})",
                )

                # Save difference visualization
                output_diff = np.abs(output_torch_vis - np.array(output_jax))
                save_samples(
                    output_diff,
                    f"test_outputs/edm/{cfg['name']}_difference.png",
                    grid_size=None,
                    colorbar=True,
                    cmap="hot",
                    title=f"Difference (Abs) ({cfg['name']})",
                )

        # Test with training mode
        if dropout > 0:
            print(f"Testing training mode with dropout={dropout}")

            # Use random sigmas for this test
            sigmas = np.random.uniform(sigma_min, sigma_max, (batch_size,))
            sigma_jax = jnp.array(sigmas)

            # Use different keys for dropout during training
            key_train = jax.random.PRNGKey(43)
            params_key_train, dropout_key_train = jax.random.split(key_train)
            rngs_train = nnx.Rngs(params=params_key_train, dropout=dropout_key_train)

            # Create a new model or update the dropout PRNG
            model_jax_train = EDMPrecond(
                rngs=rngs_train,
                img_resolution=img_resolution,
                img_channels=img_channels,
                label_dim=label_dim,
                model_type="DhariwalUNet",
                model_channels=model_channels,
                channel_mult=channel_mult,
                attn_resolutions=attn_resolutions,
                num_blocks=num_blocks,
                dropout=dropout,
                sigma_data=sigma_data,
            )

            output_jax_eval = model_jax(
                x_jax, sigma_jax, class_labels=class_labels_jax, train=False
            )
            output_jax_train = model_jax_train(
                x_jax, sigma_jax, class_labels=class_labels_jax, train=True
            )

            # Check that outputs are different when in training mode due to dropout
            train_eval_diff = np.mean(
                np.abs(np.array(output_jax_train) - np.array(output_jax_eval))
            )
            print(
                f"Mean difference between train and eval modes: {train_eval_diff:.6e}"
            )

            # We expect some difference due to dropout
            if train_eval_diff < 1e-6:
                print(
                    "WARNING: Train and eval modes produce identical outputs despite dropout"
                )
                all_passed = False

    print("\n===== EDM PRECOND TEST RESULTS =====")
    print(f"All tests passed: {all_passed}")

    return all_passed, all_results


def run_comprehensive_tests():
    """Run all comprehensive tests and summarize results"""
    print("\n===== STARTING COMPREHENSIVE MODEL TESTS =====\n")

    # Run UNet tests
    unet_passed, unet_results = test_dhariwal_unet_comprehensive()

    # Run EDM Precond tests
    edm_passed, edm_results = test_edm_precond_comprehensive()

    # Combine results
    all_results = {**unet_results, **edm_results}
    all_passed = unet_passed and edm_passed

    # Print summary
    print("\n===== TEST SUMMARY =====")
    print(f"Total tests: {len(all_results)}")
    passed_tests = sum(1 for stat in all_results.values() if stat["match"])
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(all_results) - passed_tests}")
    print(f"Overall status: {'PASSED' if all_passed else 'FAILED'}")

    # Print failures if any
    if not all_passed:
        print("\nFailed tests:")
        for name, stats in all_results.items():
            if not stats["match"]:
                print(f"  - {name}")
                print(f"    Max abs diff: {stats['max_abs_diff']:.6e}")
                print(f"    Mean abs diff: {stats['mean_abs_diff']:.6e}")

    # Create a summary visualization
    summarize_test_results(all_results)

    return all_passed, all_results


def compare_model_gradients(
    model_jax,
    model_torch,
    x_jax,
    x_torch,
    sigma_jax=None,
    sigma_torch=None,
    noise_labels_jax=None,
    noise_labels_torch=None,
    class_labels_jax=None,
    class_labels_torch=None,
):
    """
    Compare gradients between JAX and PyTorch models to ensure backpropagation works correctly.

    Args:
        model_jax: JAX model (EDMPrecond or DhariwalUNet)
        model_torch: PyTorch model (EDMPrecond or torchUnet)
        x_jax: Input tensor for JAX
        x_torch: Input tensor for PyTorch
        sigma_jax: Sigma values for JAX (for EDMPrecond)
        sigma_torch: Sigma values for PyTorch (for EDMPrecond)
        noise_labels_jax: Noise labels for JAX (for DhariwalUNet)
        noise_labels_torch: Noise labels for PyTorch (for DhariwalUNet)
        class_labels_jax: Class labels for JAX
        class_labels_torch: Class labels for PyTorch

    Returns:
        dict: Comparison results
    """
    # Determine if we're working with UNet or EDMPrecond
    is_edm = sigma_jax is not None

    # PyTorch gradient computation
    x_torch.requires_grad_(True)

    if is_edm:
        y_torch = model_torch(x_torch, sigma_torch, class_labels=class_labels_torch)
    else:
        y_torch = model_torch(
            x_torch, noise_labels_torch, class_labels=class_labels_torch
        )

    # Use mean as a scalar loss
    loss_torch = y_torch.mean()
    loss_torch.backward()

    # Get PyTorch gradients
    grad_torch = x_torch.grad.detach()

    # JAX gradient computation using jax.grad
    def loss_fn(x):
        if is_edm:
            y = model_jax(x, sigma_jax, class_labels=class_labels_jax, train=True)
        else:
            y = model_jax(
                x, noise_labels_jax, class_labels=class_labels_jax, train=True
            )
        return jnp.mean(y)

    grad_fn = jax.grad(loss_fn)
    grad_jax = grad_fn(x_jax)

    # Convert gradients for comparison
    grad_jax_nchw = grad_jax.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    # Compare gradients
    grad_stats, grad_match = compare_tensors(
        grad_torch,
        grad_jax_nchw,
        rtol=1e-2,
        atol=1e-2,  # Looser tolerances for gradients
        debug=True,
        name="Gradients",
    )

    return grad_stats


def test_gradient_computation():
    """
    Test gradient computation for both UNet and EDMPrecond models
    """
    print("\n===== TESTING GRADIENT COMPUTATION =====")

    # Create output directory
    os.makedirs("test_outputs/gradients", exist_ok=True)

    results = {}

    # Configuration for gradient test
    config = {
        "img_resolution": 32,
        "in_channels": 1,
        "out_channels": 1,
        "img_channels": 1,
        "label_dim": 0,
        "model_channels": 32,  # Smaller model for faster gradient computation
        "channel_mult": [1, 2],
        "num_blocks": 1,
        "attn_resolutions": [16],
        "dropout": 0.1,  # Use dropout to test training mode
        "sigma_data": 0.5,
        "batch_size": 2,
    }

    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    key = jax.random.PRNGKey(42)

    # Create input data
    x = np.random.randn(
        config["batch_size"],
        config["img_resolution"],
        config["img_resolution"],
        config["in_channels"],
    )
    x_jax = jnp.array(x)
    x_torch = torch.from_numpy(x).permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Test UNet gradients
    print("Testing UNet gradients...")

    noise_labels = np.random.randn(config["batch_size"])
    noise_labels_jax = jnp.array(noise_labels)
    noise_labels_torch = torch.from_numpy(noise_labels)

    # Initialize models
    output_jax, model_jax = jax_unet(
        x_jax,
        noise_labels_jax,
        None,
        config["img_resolution"],
        config["in_channels"],
        config["out_channels"],
        config["label_dim"],
        config["model_channels"],
        config["channel_mult"],
        config["num_blocks"],
        config["attn_resolutions"],
        config["dropout"],
        train=True,
        key=key,
    )

    output_torch, model_torch = torch_unet(
        x_torch,
        noise_labels_torch,
        None,
        config["img_resolution"],
        config["in_channels"],
        config["out_channels"],
        config["label_dim"],
        config["model_channels"],
        config["channel_mult"],
        config["num_blocks"],
        config["attn_resolutions"],
        config["dropout"],
    )

    # Compare gradients
    unet_grad_stats = compare_model_gradients(
        model_jax,
        model_torch,
        x_jax,
        x_torch,
        noise_labels_jax=noise_labels_jax,
        noise_labels_torch=noise_labels_torch,
    )

    results["UNet_gradients"] = unet_grad_stats

    # Test EDMPrecond gradients
    print("\nTesting EDMPrecond gradients...")

    # Create sigma values
    sigma = np.random.uniform(0.1, 10.0, (config["batch_size"],))
    sigma_jax = jnp.array(sigma)
    sigma_torch = torch.from_numpy(sigma)

    # Initialize models
    params_key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

    model_jax = EDMPrecond(
        rngs=rngs,
        img_resolution=config["img_resolution"],
        img_channels=config["img_channels"],
        label_dim=config["label_dim"],
        model_type="DhariwalUNet",
        model_channels=config["model_channels"],
        channel_mult=config["channel_mult"],
        attn_resolutions=config["attn_resolutions"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
        sigma_data=config["sigma_data"],
    )

    model_torch = torchEDMPrecond(
        img_resolution=config["img_resolution"],
        img_channels=config["img_channels"],
        label_dim=config["label_dim"],
        model_type="DhariwalUNet",
        model_channels=config["model_channels"],
        channel_mult=config["channel_mult"],
        attn_resolutions=config["attn_resolutions"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
        sigma_data=config["sigma_data"],
    )

    # Compare gradients
    edm_grad_stats = compare_model_gradients(
        model_jax,
        model_torch,
        x_jax,
        x_torch,
        sigma_jax=sigma_jax,
        sigma_torch=sigma_torch,
    )

    results["EDM_gradients"] = edm_grad_stats

    # Print summary
    print("\n===== GRADIENT TEST RESULTS =====")
    all_passed = all(stats["match"] for stats in results.values())
    print(f"All gradient tests passed: {all_passed}")

    return results


def test_edge_cases():
    """Test edge cases for both UNet and EDMPrecond models"""
    print("\n===== TESTING EDGE CASES =====")

    edge_results = {}

    # Test cases
    edge_cases = [
        {
            "name": "zero_input",
            "description": "All zeros input",
            "input_fn": lambda shape: np.zeros(shape),
        },
        {
            "name": "ones_input",
            "description": "All ones input",
            "input_fn": lambda shape: np.ones(shape),
        },
        {
            "name": "extreme_values",
            "description": "Extreme but valid values",
            "input_fn": lambda shape: np.random.uniform(-100, 100, shape),
        },
        {
            "name": "tiny_values",
            "description": "Very small but non-zero values",
            "input_fn": lambda shape: np.random.uniform(1e-10, 1e-8, shape),
        },
    ]

    # Basic configuration
    config = {
        "img_resolution": 32,
        "in_channels": 1,
        "out_channels": 1,
        "img_channels": 1,
        "label_dim": 0,
        "model_channels": 32,
        "channel_mult": [1, 2, 2],
        "num_blocks": 1,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "sigma_data": 0.5,
        "batch_size": 2,
    }

    for case in edge_cases:
        print(f"\nTesting edge case: {case['name']} - {case['description']}")

        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        key = jax.random.PRNGKey(42)

        # Generate input according to the edge case
        shape = (
            config["batch_size"],
            config["img_resolution"],
            config["img_resolution"],
            config["in_channels"],
        )
        x = case["input_fn"](shape)
        x_jax = jnp.array(x)
        x_torch = torch.from_numpy(x).permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Test UNet
        print(f"Testing UNet with {case['name']}...")

        noise_labels = np.random.randn(config["batch_size"])
        noise_labels_jax = jnp.array(noise_labels)
        noise_labels_torch = torch.from_numpy(noise_labels)

        # Forward pass in JAX
        output_jax, _ = jax_unet(
            x_jax,
            noise_labels_jax,
            None,
            config["img_resolution"],
            config["in_channels"],
            config["out_channels"],
            config["label_dim"],
            config["model_channels"],
            config["channel_mult"],
            config["num_blocks"],
            config["attn_resolutions"],
            config["dropout"],
            train=False,
            key=key,
        )

        # Forward pass in PyTorch
        output_torch, _ = torch_unet(
            x_torch,
            noise_labels_torch,
            None,
            config["img_resolution"],
            config["in_channels"],
            config["out_channels"],
            config["label_dim"],
            config["model_channels"],
            config["channel_mult"],
            config["num_blocks"],
            config["attn_resolutions"],
            config["dropout"],
        )

        # Convert JAX output to match PyTorch format for comparison
        output_jax_nchw = output_jax.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # Compare outputs
        stats, passed = compare_tensors(
            output_torch,
            output_jax_nchw,
            rtol=1e-3,
            atol=1e-3,
            debug=True,
            name=f"UNet-{case['name']}",
        )

        edge_results[f"UNet-{case['name']}"] = stats

        # Test EDMPrecond
        print(f"Testing EDMPrecond with {case['name']}...")

        # Create sigma values - use the same for all edge cases
        sigma = np.random.uniform(0.1, 10.0, (config["batch_size"],))
        sigma_jax = jnp.array(sigma)
        sigma_torch = torch.from_numpy(sigma)

        # Initialize models
        params_key, dropout_key = jax.random.split(key)
        rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

        model_jax = EDMPrecond(
            rngs=rngs,
            img_resolution=config["img_resolution"],
            img_channels=config["img_channels"],
            label_dim=config["label_dim"],
            model_type="DhariwalUNet",
            model_channels=config["model_channels"],
            channel_mult=config["channel_mult"],
            attn_resolutions=config["attn_resolutions"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
            sigma_data=config["sigma_data"],
        )

        model_torch = torchEDMPrecond(
            img_resolution=config["img_resolution"],
            img_channels=config["img_channels"],
            label_dim=config["label_dim"],
            model_type="DhariwalUNet",
            model_channels=config["model_channels"],
            channel_mult=config["channel_mult"],
            attn_resolutions=config["attn_resolutions"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
            sigma_data=config["sigma_data"],
        )

        # Forward pass
        output_jax = model_jax(x_jax, sigma_jax, class_labels=None, train=False)
        output_torch = model_torch(x_torch, sigma_torch, class_labels=None)

        # Convert JAX output to match PyTorch format for comparison
        output_jax_nchw = output_jax.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # Compare outputs
        stats, passed = compare_tensors(
            output_torch,
            output_jax_nchw,
            rtol=1e-3,
            atol=1e-3,
            debug=True,
            name=f"EDM-{case['name']}",
        )

        edge_results[f"EDM-{case['name']}"] = stats

    # Print summary
    print("\n===== EDGE CASE TEST RESULTS =====")
    all_passed = all(stats["match"] for stats in edge_results.values())
    print(f"All edge case tests passed: {all_passed}")

    return edge_results


def summarize_test_results(results):
    """Create a summary visualization of all test results"""
    # Prepare data for plotting
    test_names = list(results.keys())
    max_diffs = [results[name]["max_abs_diff"] for name in test_names]
    mean_diffs = [results[name]["mean_abs_diff"] for name in test_names]
    passed = [results[name]["match"] for name in test_names]

    # Sort by category and then by max difference
    categories = [name.split("-")[0] for name in test_names]
    sorted_indices = sorted(
        range(len(test_names)), key=lambda i: (categories[i], -max_diffs[i])
    )

    test_names = [test_names[i] for i in sorted_indices]
    max_diffs = [max_diffs[i] for i in sorted_indices]
    mean_diffs = [mean_diffs[i] for i in sorted_indices]
    passed = [passed[i] for i in sorted_indices]

    # Truncate long names
    display_names = [
        name[:30] + "..." if len(name) > 30 else name for name in test_names
    ]

    # Create bar chart
    plt.figure(figsize=(12, max(8, len(test_names) * 0.4)))

    # Use log scale for differences
    bars = plt.barh(
        range(len(test_names)), max_diffs, color=["g" if p else "r" for p in passed]
    )

    # Add mean diff as a dot
    for i, mean_diff in enumerate(mean_diffs):
        plt.plot(mean_diff, i, "ko", markersize=5)

    plt.yscale("linear")
    plt.xscale("log")
    plt.xlabel("Maximum Absolute Difference (log scale)")
    plt.ylabel("Test Case")
    plt.yticks(range(len(display_names)), display_names)
    plt.title("Model Test Results - Maximum Differences")
    plt.grid(axis="x", which="both", linestyle="--", alpha=0.7)

    # Add a legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="g", lw=4, label="Passed"),
        Line2D([0], [0], color="r", lw=4, label="Failed"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            label="Mean Difference",
            markersize=8,
            linestyle="",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # Add threshold line
    plt.axvline(x=1e-3, color="blue", linestyle="--", alpha=0.7)
    plt.text(
        1.1e-3,
        len(test_names) - 1,
        "Threshold (1e-3)",
        rotation=90,
        va="top",
        color="blue",
    )

    plt.tight_layout()
    plt.savefig("test_outputs/test_summary.png", dpi=150)
    plt.close()

    print("Test summary visualization saved to test_outputs/test_summary.png")


if __name__ == "__main__":
    # Run comprehensive tests
    run_comprehensive_tests()

    # Test gradient computation
    test_gradient_computation()

    # Test edge cases
    test_edge_cases()
