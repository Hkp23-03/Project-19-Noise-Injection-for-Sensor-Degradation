# Project 19: Noise Injection for Sensor Degradation

## Objective
A Staff-Level semantic segmentation project tailored for real-world Unmanned Ground Vehicle (UGV) operations. This project utilizes a DeepLabV3 architecture and injects synthetic Gaussian and Salt/Pepper noise during training to simulate camera degradation such as dust and static. The goal is to ensure less than a 5% IoU drop when encountering these conditions in production.

## Environment Setup
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python tqdm scikit-learn
```

## Dataset Structure
Place your dataset in the `data/` directory following this exact format:
```text
data/
├── train/
│   ├── images/  (RGB Images)
│   └── masks/   (1-Channel Grayscale Label Maps, indices 0-9)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Training
To begin training the DeepLabV3 model with noise augmentations:
```bash
python train.py --epochs 50
```
The dataloader utilizes `albumentations` for `GaussNoise` and `PixelDropout` (Salt/Pepper equivalent) on-the-fly to ensure the model develops robustness to sensor interference. 

## Inference & Failure Analysis
To test the inference speed, extract metrics, and view failure analyses:
```bash
python test.py
```
This script runs validation on NOISY test images to verify the resilience of the network and logs specific failure cases or IoU drops resulting from the noise.
