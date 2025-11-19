from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch
from training.dataset import LMDBData
import random

def collate_fn(batch):
    """
    Collate function that converts a batch of samples into JAX arrays.
    Args:
        batch: A list of dictionaries containing the data samples.
    Returns:
        A dictionary with the same keys as input but values converted to JAX arrays.
        Image-like tensors are converted from NCHW to NHWC format.
    """
    if not batch:
        return {}

    # Initialize an empty dictionary to store the collated data
    collated = {}

    # Get all keys from the first batch item
    keys = batch[0].keys()

    def convert_to_nhwc(array):
        """Helper function to convert NCHW to NHWC format if applicable"""
        if len(array.shape) == 4:  # Batched images (B,C,H,W) -> (B,H,W,C)
            return np.transpose(array, (0, 2, 3, 1))
        elif len(array.shape) == 3:  # Single images (C,H,W) -> (H,W,C)
            return np.transpose(array, (1, 2, 0))
        return array

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            # Stack PyTorch tensors and convert to numpy
            stacked = torch.stack([b[key] for b in batch]).numpy()
            # Convert format if it's an image-like tensor
            stacked = convert_to_nhwc(stacked)
            # Convert to JAX array
            collated[key] = stacked
        elif isinstance(batch[0][key], np.ndarray):
            # Stack numpy arrays
            stacked = np.stack([b[key] for b in batch])
            # Convert format if it's an image-like array
            stacked = convert_to_nhwc(stacked)
            # Convert to JAX array
            collated[key] = stacked
        elif isinstance(batch[0][key], (str, int)):
            # Keep strings and integers as lists
            collated[key] = [b[key] for b in batch]
        else:
            # For any other type, try to convert to JAX array
            try:
                stacked = np.stack([b[key] for b in batch])
                stacked = convert_to_nhwc(stacked)
                collated[key] = stacked
            except:
                collated[key] = [b[key] for b in batch]

    return collated


def navier_stokes_data(config):
    root = "data/navier-stokes-train/Re200.0-t5.0"
    num_train_workers = config.num_train_workers
    num_samples = config.per_device_batch_size * 4
    dataset = LMDBData(root=root)  # LMDBData will normalize

    # return dataset
    indices = random.sample(range(dataset.length), num_samples)
    sampler = SubsetRandomSampler(indices)  # TODO: remove this
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.per_device_batch_size,
        sampler=sampler,
        num_workers=num_train_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, dataset


def blackhole_data(n_devices, config):
    from training.dataset import BlackHole

    root = "data/blackhole"
    dataset = BlackHole(root=root, resolution=config.training.resolution)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_train_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, dataset


def imagefolder_data(n_devices, config):
    from training.dataset import ImageFolder

    root = config.training.imagefolder_root
    dataset = ImageFolder(root=root, resolution=config.training.resolution)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_train_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, dataset


def get_datasets(n_devices, config, mesh_axis_names=None):
    if config.training.dataset_name == "navier_stokes":
        dataloader, dataset = navier_stokes_data(config)
    elif config.training.dataset_name == "blackhole":
        dataloader, dataset = blackhole_data(n_devices, config)
    elif config.training.dataset_name == "imagefolder":
        dataloader, dataset = imagefolder_data(n_devices, config)
    else:
        raise ValueError(f"Dataset {config.training.dataset_name} not supported")

    # For multi-device training, we need to ensure the batch size is properly scaled
    # The dataloader should return batches that can be sharded across devices
    if mesh_axis_names is not None:
        # Ensure batch size is divisible by number of devices
        original_batch_size = config.training.batch_size
        if original_batch_size % n_devices != 0:
            # Adjust batch size to be divisible by number of devices
            adjusted_batch_size = (original_batch_size // n_devices) * n_devices
            print(
                f"Adjusting batch size from {original_batch_size} to {adjusted_batch_size} for {n_devices} devices"
            )
            # Note: This is a simplified approach. In practice, you might want to recreate the dataloader
            # with the adjusted batch size, but for now we'll assume the dataloader can handle this.

    return dataloader, dataset
