from training.dataset import LMDBData
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Use the same root as in nnx_train.py
    root = "data/navier-stokes-train/Re200.0-t5.0"
    batch_size = 4
    dataset = LMDBData(root=root, norm=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Get one batch
    for batch in dataloader:
        images = batch["target"]
        print(f"Batch shape: {images.shape}")
        print(f"Max value: {images.max().item()}")
        print(f"Min value: {images.min().item()}")
        break
