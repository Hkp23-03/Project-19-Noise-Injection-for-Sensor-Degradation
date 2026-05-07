import os
import torch

# General Project Configuration
PROJECT_NAME = "Noise_Injection_Sensor_Degradation"
SEED = 42

# Dataset Configuration
DATA_DIR = "./data"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

NUM_CLASSES = 10
CLASS_NAMES = [
    "Background", "Terrain", "Vegetation", "Sky", "Obstacle", 
    "Logs", "Water", "Vehicle", "Person", "Animal"
]

# Model Configuration
MODEL_BACKBONE = "resnet50"
IN_CHANNELS = 3
OUT_CHANNELS = NUM_CLASSES

# Training Configuration
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Hardware Toggles
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True # Automatic Mixed Precision
NUM_WORKERS = 4

# Target Constraints
TARGET_IOU = 80.0
INFERENCE_SPEED_MS = 50.0

# Paths
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
