import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

# Configure logger
logger = logging.getLogger(__name__)

class SegmentationDataset(Dataset):
    """
    Robust PyTorch Dataset for Semantic Segmentation.
    Handles corrupted files gracefully by skipping or substituting them.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Verify pairs exist to prevent run-time errors
        self.valid_images = []
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            for img_name in os.listdir(image_dir):
                if os.path.exists(os.path.join(mask_dir, img_name)):
                    self.valid_images.append(img_name)
                else:
                    logger.warning(f"Missing mask for image: {img_name}. Skipping.")
        else:
            logger.warning(f"Data directory not found. Please setup: {image_dir}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, index):
        img_name = self.valid_images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Assumes mask is 1-channel with class indices
            
            if image is None or mask is None:
                raise ValueError("Corrupted image or mask")
        except Exception as e:
            logger.error(f"Failed to load image/mask pair {img_name}: {str(e)}")
            # Fallback to another index
            return self.__getitem__((index - 1) % max(1, len(self.valid_images)))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask.long()

def get_train_transforms():
    """
    Training transformations injecting Gaussian and Salt/Pepper noise.
    Simulates real-world UGV camera degradation like dust and static.
    """
    return A.Compose(
        [
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            # Noise Injection: Gaussian Noise
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
            # Noise Injection: Salt/Pepper equivalent (PixelDropout sets random pixels to black/0)
            A.PixelDropout(dropout_prob=0.05, per_channel=False, drop_value=0, p=0.5),
            # Simulate bright dust specs (Salt)
            A.PixelDropout(dropout_prob=0.02, per_channel=False, drop_value=255, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

def get_val_transforms():
    """Validation transformations (Includes noise to test the constraint of <5% IoU drop)."""
    return A.Compose(
        [
            A.Resize(height=512, width=512),
            # Injecting noise into validation to measure robust IoU
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.PixelDropout(dropout_prob=0.05, drop_value=0, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

def create_dataloaders(train_img, train_mask, val_img, val_mask, batch_size, num_workers):
    train_dataset = SegmentationDataset(train_img, train_mask, transform=get_train_transforms())
    val_dataset = SegmentationDataset(val_img, val_mask, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
