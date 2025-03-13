import os
import torch
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Class index mapping
GLEASON_CLASSES = {"healthy": 0, "gleason-3": 1, "gleason-4": 2, "gleason-5": 3}
# TODO: Might want to group into "clinically significant: yes/no" (healthy+3 vs. 4+5)

class ProstateMRISegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        """
        Args:
            image_dir (str): Path to images.
            mask_dir (str): Path to corresponding masks.
            transform (callable, optional): Transformations for images.
            target_transform (callable, optional): Transformations for masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_filenames = sorted(os.listdir(image_dir))  # Ensure order consistency

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format

        # Find all masks corresponding to this image
        base_name = image_filename.replace(".png", "")
        mask_paths = glob.glob(os.path.join(self.mask_dir, f"{base_name}_*.png"))

        # Initialize segmentation mask (single-channel, filled with background class 0)
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)  # (H, W)

        for mask_path in mask_paths:
            mask_label = mask_path.split("_")[-1].replace(".png", "")  # Extract label
            if mask_label in GLEASON_CLASSES:
                mask_class = GLEASON_CLASSES[mask_label]  # Convert to class index
                mask_layer = Image.open(mask_path).convert("L")  # Load as grayscale
                
                # Convert to NumPy array
                mask_layer = np.array(mask_layer)
                mask_layer[mask_layer > 0] = mask_class  # Assign class index to foreground

                # Merge overlapping regions by taking the max class value
                mask = np.maximum(mask, mask_layer)

        # Convert mask to tensor (single-channel)
        mask = torch.tensor(mask, dtype=torch.long)  # Shape: (H, W), values in {0,1,2,3}

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
