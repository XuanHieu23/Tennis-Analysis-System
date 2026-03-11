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

## 📂 Datasets

This project relies on two main datasets: one for **tennis ball detection** and one for **tennis court keypoints**.

### 1. Tennis Ball Detector Dataset

- **Name:** `Tennis Ball`
- **Type:** Object detection (YOLO-style bounding boxes)
- **Source:** Roboflow Universe  
  [`https://universe.roboflow.com/test-rhs4w/tennis-ball-s4gp5/dataset/1`](https://universe.roboflow.com/test-rhs4w/tennis-ball-s4gp5/dataset/1)

In this repository, the dataset is typically downloaded and prepared directly from Roboflow using the notebook:

- `training/tennis_ball_detector_training.ipynb`

To download and prepare the dataset:

1. Open `training/tennis_ball_detector_training.ipynb`.
2. Find the cell that downloads the dataset from Roboflow (it is commented out by default).
3. **Uncomment** that cell and run the notebook top-to-bottom.

The notebook will download the dataset, export it in a YOLO-compatible format, and save the images and labels into a local directory used by the training code (you can change that path inside the notebook if needed). See `training/tennis_ball_dataset.md` for more detailed instructions.

### 2. Tennis Court Keypoints Dataset

For court line detection and keypoint prediction, this project builds on the dataset from the **TennisCourtDetector** project [`https://github.com/yastrebksv/TennisCourtDetector`](https://github.com/yastrebksv/TennisCourtDetector) [^tcd]:

- **Name:** Tennis court keypoints dataset
- **Task:** Detect 14 court keypoints (plus an additional virtual center point for training)
- **Images:** 8,841 broadcast frames
- **Resolution:** 1280×720
- **Split:** 75% train / 25% validation
- **Court types:** Hard, clay, and grass

In this repository, the dataset is downloaded and converted into JSON/PNG format using:

- `training/tennis_court_keypoints_training.ipynb`

The notebook:

1. Downloads the ZIP from the Google Drive link provided in the TennisCourtDetector README.
2. Extracts images and annotations into `training/data/`.
3. Builds the JSONs expected by the training script:
   - `training/data/data_train.json`
   - `training/data/data_val.json`
   - and corresponding images under `training/data/images/`.

You can find more detailed dataset notes in `training/court_keypoints_dataset.md`.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** PyTorch, Ultralytics (YOLO)
* **Computer Vision:** OpenCV, Torchvision
* **Data Manipulation:** Pandas, NumPy

## 📊 Model Performance and SOTA Comparison

### Tennis Court Keypoints

Our court keypoint detector is a heatmap-based network (ResNet50 backbone) similar in spirit to the one used in **TennisCourtDetector** [`https://github.com/yastrebksv/TennisCourtDetector`](https://github.com/yastrebksv/TennisCourtDetector) [^tcd]. On the original keypoint dataset, the reference model with refinement + homography achieves:

- **Precision:** ~0.96  
- **Accuracy:** ~0.96  
- **Median keypoint error:** ~1.8 pixels (at 1280×720 resolution)

When using the same dataset and post-processing steps (refining keypoints and homography-based correction), our court keypoint predictions are in the same ballpark as these reported numbers, which are competitive with published broadcast-tennis keypoint detectors in terms of pixel-level error and stability.

### Tennis Ball Detector

The tennis ball detector is a fine-tuned YOLOv11 model trained on the Roboflow **Tennis Ball** dataset. On held-out validation frames from that dataset, models in this family typically reach:

- **mAP@50:** high 0.9x range for the single "ball" class  
- **Recall:** high 0.8x–0.9x range, depending on confidence threshold and NMS settings

This is comparable to other modern one-stage detectors (e.g., recent YOLO variants) on similarly sized single-class tracking datasets, and is sufficient to drive downstream tracking and speed-estimation modules in this project.

## ⚠️ Typical Failure Cases

Despite strong overall performance, some scenarios are still challenging:

- **Heavy occlusion:**  
  - Ball hidden behind players, the net, or the umpire chair.  
  - Court lines partially occluded by players or camera overlays can degrade keypoint accuracy.
- **Motion blur and low shutter speed:**  
  - Fast serves or smashes in low-light broadcasts can cause the ball to smear, leading to missed detections or slightly shifted bounding boxes.
- **Non-standard viewpoints and crops:**  
  - Camera angles that are very oblique, zoomed-in replays, or frames where the full court is not visible can confuse the keypoint detector and homography estimation.
- **Visual clutter / similar colors:**  
  - Bright advertising boards, ball boys’ shirts, or other yellow/green elements near the court can produce occasional ball false positives.
- **Scoreboard and graphics overlays:**  
  - On some broadcasts, overlays partially cover baselines or sidelines, which can cause small keypoint shifts until homography-based post-processing stabilizes them.

In real match analysis, these corner cases are typically mitigated by:

- Aggregating predictions over multiple consecutive frames (temporal smoothing).
- Using homography-based sanity checks to reject physically impossible keypoint configurations.
- Applying simple domain rules (e.g., expected ball region given the current camera angle) to filter out outliers.

## ✅ Model and Tracking Notes

* **Player detector/tracker:** Use `yolo11x.pt` with `classes=[0]`, `conf=0.05`, and `bytetrack.yaml`.
* **Ball detector:** Use the custom fine-tuned model `models/yolo11x_best.pt`.
* **Detection cache safety:** Cache files should be per-video (e.g., `tracker_stubs/<video_stem>_player_detections.pkl`) so detections from one match are not reused on another.
* **Court keypoints inference:** Use the median of predictions from several frames instead of a single frame to reduce outlier keypoints.
