## 🎾 Tennis Ball Detector Dataset

This project uses the **"Tennis Ball"** dataset hosted on Roboflow Universe:

- **Dataset name:** `Tennis Ball`
- **Source:** [Roboflow Universe – Tennis Ball Detector](https://universe.roboflow.com/test-rhs4w/tennis-ball-s4gp5/dataset/1)
- **Task:** Tennis ball detection (object detection)

The dataset contains broadcast tennis frames with bounding boxes around the ball. It is already split into training/validation/test sets on Roboflow.

### How to download the dataset in this repo

The easiest way to download and prepare this dataset for training is to use the provided notebook:

- `training/tennis_ball_detector_training.ipynb`

Steps:

1. Open `tennis_ball_detector_training.ipynb`.
2. Find the cell that downloads the dataset from Roboflow (it is intentionally **commented out**).
3. **Uncomment** the download code in that cell.
4. Run the notebook top-to-bottom.

The notebook will:

- Authenticate with Roboflow (you may need to paste your Roboflow API key).
- Download the tennis ball dataset.
- Export it into a YOLO-friendly format.
- Save the images and labels into a local directory used later in the training code (e.g. `training/ball_data/` – you can change this path inside the notebook).

If you prefer not to use the notebook, you can also:

1. Visit the dataset page on Roboflow.
2. Download it manually in YOLO format.
3. Extract it into the same directory expected by the notebook / training script.
