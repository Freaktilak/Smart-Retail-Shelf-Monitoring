# Smart Retail Shelf Monitoring System
## DATA 690 - Applied Artificial Intelligence | UMBC | Spring 2025

## Problem Statement
Retail stores lose billions annually due to out-of-stock shelves.
This system uses computer vision to automatically detect missing
or low-stock shelf regions in real time using a Jetson Nano 
edge device.

## Running Model
- Model: YOLOv8s (Small variant)
- Format: ONNX (opset 19) for Jetson Nano compatibility
- Runtime: ONNX Runtime 1.16.3
- Device: NVIDIA Jetson Nano (JetPack 5.1.6)

## Weights
- Base weights: YOLOv8s pretrained on COCO dataset
- Fine-tuned on: SKU110K retail shelf dataset
- Training: Google Colab Pro (Tesla T4 GPU)
- Epochs: 15
- Training time: 1.92 hours
- Weight file: best_opset19.onnx (43MB)

## Inference
- Input: Live USB camera feed (640x480)
- Processing: Real-time YOLO detection + custom grid logic
- Output: Bounding boxes + empty zone highlighting + dashboard

## Predictions
- Green boxes: Detected products on shelf
- Red zones: Empty/missing stock areas
- Grid: 4x6 = 24 shelf zones analyzed per frame
- Alert: Automatic restock warning when empty zones > 3

## Speed (Performance)
- FPS on Jetson Nano: ~4.3 FPS (CPU inference)
- Frame skipping: Every 2nd frame for smoother display
- Inference time: ~230ms per frame on Jetson Nano CPU

## Metrics
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.882 |
| mAP@0.5:0.95 | 0.528 |
| Precision | 0.886 |
| Recall | 0.823 |
| FPS (Jetson Nano) | ~4.3 |

## Transfer Learning Steps
1. Load YOLOv8s pretrained on COCO (80 classes)
2. Freeze first 10 layers (general features preserved)
3. Fine-tune remaining layers on SKU110K shelf images
4. Export to ONNX format for edge deployment
5. Deploy on Jetson Nano with ONNX Runtime

## Dataset
- SKU110K: 11,762 retail shelf images
- Training set: 6,998 images
- Validation set: 2,000 images
- Bounding box annotations included

## Tech Stack
- Python 3.8
- YOLOv8 (Ultralytics)
- ONNX Runtime 1.16.3
- OpenCV 4.2.0
- NumPy 1.24.4
- Jetson Nano (Ubuntu 20.04, JetPack 5.1.6)

## How to Run
```bash
cd SmartShelf
python3 smart_shelf.py
```

## Controls
| Key | Action |
|-----|--------|
| Z / X | Zoom In / Out |
| W / V | Pan Up / Down |
| A / D | Pan Left / Right |
| R | Reset View |
| S | Save Screenshot |
| Q | Quit |
