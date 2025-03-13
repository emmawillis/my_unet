import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # For progress bar
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import ProstateMRISegmentationDataset
from unet import UNet

# Device configuration (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standardize intensities (see utils/analyze_dataset.py)
DATASET_PIXEL_MEAN = 0.1726
DATASET_PIXEL_STD = 0.1411

def validate_unet(model, val_loader, loss_fn):
    model.eval()  # Set to evaluation mode
    val_loss = 0

    with torch.no_grad():  # Disable gradient computation
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, masks.squeeze(1))
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    model.train()  # Switch back to training mode
    return val_loss / len(val_loader)

def train_unet(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20):
    model.train()  # Set to training mode

    val_loss = 0
    best_loss = float("inf")
    patience = 10  # Stop training if no improvement for 10 epochs
    patience_counter = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(images)  # Output shape: (batch, 4, 256, 256)

            # Compute loss (CrossEntropyLoss expects class labels, not one-hot masks)
            loss = loss_fn(outputs, masks.squeeze(1))  # `masks` should have shape (batch, 256, 256) with values 0-3

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        # Run validation every few epochs
        if (epoch + 1) % 5 == 0:
            val_loss += validate_unet(model, val_loader, loss_fn)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

            # Reduce learning rate if validation loss stops improving
            scheduler.step(val_loss)

    print("Training Complete!")


def train():
    print("device IN submitit: ", device)

    # Initialize the model
    model = UNet(in_channels=3, out_channels=4).to(device)  # 4 output channels for multi-class

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Suitable for multi-class masks
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mri_transform = transforms.Compose([
        transforms.Normalize(mean=[DATASET_PIXEL_MEAN], std=[DATASET_PIXEL_STD])
    ])

    train_dataset = ProstateMRISegmentationDataset('dataset_split/train/images', 'dataset_split/train/images', transform=mri_transform)
    test_dataset = ProstateMRISegmentationDataset('dataset_split/test/images', 'dataset_split/test/images', transform=mri_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    train_unet(model, train_loader, test_loader, optimizer, loss_fn, num_epochs=20)

