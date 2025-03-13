import os
import torch
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

GLEASON_CLASSES = {"healthy": 0, "gleason-3": 1, "gleason-4": 2, "gleason-5": 3}
 #TODO might want to just do clincially significant yes / no (ie healthy/3 and 4/5)

class ProstateMRISegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        """
        Args:
            image_dir (str): Path to images.
            mask_dir (str): Path to corresponding masks.
            transform (callable, optional): Transformation for images.
            target_transform (callable, optional): Transformation for masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

        # Get all image filenames (sorted for consistency)
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

def __getitem__(self, idx):
        # Load image
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Find all corresponding masks for this image
        base_name = image_filename.replace(".png", "")
        mask_paths = glob.glob(os.path.join(self.mask_dir, f"{base_name}_*.png"))

        # Initialize segmentation mask (single-channel, same size as image, filled with background class 0)
        mask = None
        for mask_path in mask_paths:
            mask_label = mask_path.split("_")[-1].replace(".png", "")  # Extract class label
            if mask_label in GLEASON_CLASSES:
                mask_class = GLEASON_CLASSES[mask_label]  # Convert label to class index
                mask_layer = Image.open(mask_path).convert("L")  # Load as grayscale

                # Convert to NumPy array and assign class index where the mask is nonzero
                mask_layer = np.array(mask_layer)
                mask_layer[mask_layer > 0] = mask_class  # Assign class index to foreground pixels

                # Combine with existing mask (assign highest class in case of overlap)
                if mask is None:
                    mask = mask_layer
                else:
                    mask = np.maximum(mask, mask_layer)  # Take max class in overlapping regions

        # If no mask exists, create an empty mask (background class 0)
        if mask is None:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)  # (H, W)

        # Convert mask to tensor (single-channel)
        mask = torch.tensor(mask, dtype=torch.long)  # Shape: (H, W), values are class indices

        # Apply transformations (resize, normalize)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask