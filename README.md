# 🎾 Tennis Analysis System

A comprehensive Computer Vision project designed to analyze tennis match videos. This system leverages deep learning models to track players and the ball, detect court keypoints, and extract real-time match statistics such as shot speed and player movement.

## ✨ Key Features

* **Player & Ball Tracking:** Utilizes YOLO models (`yolov11x` for players and a custom YOLOv11 model for the ball) to accurately detect and track objects across frames.
* **Court Line Detection:** Employs a custom-trained ResNet50 model to predict 14 crucial court keypoints, ensuring accurate spatial awareness.
* **2D Mini-Court Projection:** Maps the real-world coordinates of players and the ball onto a 2D top-down mini-court representation using perspective transformation concepts.
* **Advanced Statistical Analysis:** * Automatically detects the exact frames where ball hits occur.
    * Calculates player movement speed (km/h).
    * Calculates ball shot speed (km/h).
    * Displays real-time and average stats on a clean, dynamic UI overlay.
* **Missing Data Interpolation:** Uses Pandas to smoothly interpolate missing ball detections and prevent tracking jitter.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** PyTorch, Ultralytics (YOLO)
* **Computer Vision:** OpenCV, Torchvision
* **Data Manipulation:** Pandas, NumPy

## ✅ Model and Tracking Notes

* **Player detector/tracker:** Use `yolo11x.pt` with `classes=[0]`, `conf=0.15`, and `bytetrack.yaml`.
* **Ball detector:** Use the custom fine-tuned model `models/yolo11x_best.pt`.
* **Detection cache safety:** Cache files should be per-video (e.g., `tracker_stubs/<video_stem>_player_detections.pkl`) so detections from one match are not reused on another.
* **Court keypoints inference:** Use the median of predictions from several frames instead of a single frame to reduce outlier keypoints.

## 🏋️ Keypoint Training

Use the script below (instead of the notebook-only flow) to train with validation, best-checkpoint saving, and early stopping:

```bash
python3 training/train_court_keypoints.py \
  --images-dir training/data/images \
  --train-json training/data/data_train.json \
  --val-json training/data/data_val.json \
  --output models/keypoints_model.pth
```
