import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader import ProstateMRISegmentationDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def compute_mean_std():
    # Load dataset (without normalization)
    mri_dataset = ProstateMRISegmentationDataset(
        image_dir="dataset_split/train/images",
        mask_dir="dataset_split/train/masks",
    )
    loader = DataLoader(mri_dataset, batch_size=16, shuffle=False, num_workers=4)
    mean, std = 0.0, 0.0
    num_pixels = 0

    for images, _ in loader:
        images = images.view(images.size(0), -1)  # Flatten each image
        mean += images.mean(dim=1).sum()
        std += images.std(dim=1).sum()
        num_pixels += images.size(0)  # Batch size

    mean /= num_pixels
    std /= num_pixels
    print(f"Computed Mean: {mean.item()}, Computed Std: {std.item()}")
    return mean.item(), std.item()


compute_mean_std()