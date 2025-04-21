from training.dataset import LMDBData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    exp_dir = os.path.join("exps/pretrain", "LMBDtest")
    os.makedirs(exp_dir, exist_ok=True)
    root = "data/navier-stokes/Re200.0-t5.0"
    num_train_workers = 4
    batch_size = 4

    dataset = LMDBData(root=root)

    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_train_workers,
            pin_memory=True,
            drop_last=True,
        )
    num_epochs = 5
    train_steps = 0
    max_steps = 200
    for e in range(num_epochs):
        for imgs in dataloader:
            print(f"train_steps: {train_steps}")
            if train_steps >= max_steps: break
            
            if train_steps % 20 == 0:
                print(f"here: {train_steps}")
                sample_path = os.path.join(exp_dir, f"samples_step_{train_steps}.png")
                # print(imgs.shape)
                print(type(imgs))
                print(imgs[0])
                save_samples(imgs, sample_path)
            train_steps+=1
        print(f"Epoch: {e}")
                

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
    print(samples_np.shape)
    
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


if __name__ == "__main__":
    main()