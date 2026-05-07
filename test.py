import argparse
import time
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import logging

import config
from dataset import create_dataloaders
from model import DeepLabV3PlusModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(conf_matrix):
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (tp + fp + fn)
        recall = tp / (tp + fn)

    return iou, recall

def main():
    parser = argparse.ArgumentParser(description="Test Inference under simulated sensor degradation")
    args = parser.parse_args()

    device = config.DEVICE
    logger.info(f"Testing on device: {device}")

    # Use validation transforms which include the Noise injection
    _, test_loader = create_dataloaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        batch_size=1, num_workers=1
    )

    if len(test_loader) == 0:
        logger.error("Test dataloader empty.")
        return

    model = DeepLabV3PlusModel(num_classes=config.NUM_CLASSES, pretrained_backbone=False).to(device)
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
        logger.info("Loaded best model weights.")
    else:
        logger.warning("No pretrained weights found! Initializing randomly.")

    model.eval()
    total_time = 0.0
    num_images = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(images)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000 # ms
            total_time += inference_time
            num_images += 1
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

    avg_time_ms = total_time / num_images
    logger.info(f"Average Inference Time per image: {avg_time_ms:.2f} ms")
    
    if avg_time_ms > config.INFERENCE_SPEED_MS:
        logger.warning(f"Inference speed constraint failed: {avg_time_ms:.2f}ms > {config.INFERENCE_SPEED_MS}ms")
    else:
        logger.info(f"Inference speed constraint met! (< {config.INFERENCE_SPEED_MS}ms)")

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(config.NUM_CLASSES)))
    ious, recalls = calculate_metrics(cm)
    mean_iou = np.nanmean(ious)
    
    logger.info(f"Final Mean IoU on NOISY data: {mean_iou * 100:.2f}%")
    
    # Analyze Failure Cases
    logger.info("\n--- Failure Case Analysis ---")
    logger.info("The model was tested heavily on Gaussian and Salt/Pepper noise (simulating dust/static).")
    if mean_iou * 100 > config.TARGET_IOU:
        logger.info("SUCCESS: The model handled degradation robustly and maintained IoU above the target.")
    else:
        logger.warning(f"DEGRADATION IMPACT: IoU dropped to {mean_iou * 100:.2f}%. Noise likely caused high false positives in boundary regions or fine details.")

    logger.info("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
