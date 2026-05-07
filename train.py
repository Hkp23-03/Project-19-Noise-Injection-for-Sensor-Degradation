import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import DeepLabV3PlusModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_iou(preds, labels, num_classes):
    """Calculate mean IoU for all classes."""
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.nanmean(ious)

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    epoch_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    total_iou = 0.0
    batches = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            total_iou += calculate_iou(outputs, masks, num_classes)
            batches += 1
            
    return epoch_loss / len(loader), total_iou / batches

def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 with Noise Injection")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()

    set_seed(config.SEED)
    logger.info(f"Starting Project: {config.PROJECT_NAME}")
    
    device = config.DEVICE
    logger.info(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        logger.error("Dataloaders empty. Populate data/")
        return

    model = DeepLabV3PlusModel(num_classes=config.NUM_CLASSES, pretrained_backbone=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_iou = val_epoch(model, val_loader, criterion, device, config.NUM_CLASSES)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            logger.info(f"Saved new best model with mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
