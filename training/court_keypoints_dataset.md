## 🏟️ Tennis Court Keypoints Dataset

For court line and keypoint detection, this project builds on the dataset released with the **TennisCourtDetector** project:

- **Dataset name:** Tennis court keypoints dataset (14 keypoints + court center)
- **Source repo:** [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
- **Dataset download (Google Drive):** Link in the TennisCourtDetector README  
  (see the "Dataset" section: training images + 14 annotated court keypoints) [`https://github.com/yastrebksv/TennisCourtDetector`](https://github.com/yastrebksv/TennisCourtDetector)

From the original description:

- **Images:** 8,841 frames from broadcast tennis videos.
- **Resolution:** 1280×720.
- **Annotations:** 14 court keypoints per image (plus an additional virtual center point in the model).
- **Split:** 75% training, 25% validation.
- **Court types:** Hard, clay, and grass.

### How to download and prepare the dataset in this repo

The recommended way to download and structure the court keypoints dataset for this project is via the notebook:

- `training/tennis_court_keypoints_training.ipynb`

Steps:

1. Open `tennis_court_keypoints_training.ipynb`.
2. Locate the cell that **downloads and/or unpacks** the Google Drive dataset (it is intentionally **commented out**).
3. **Uncomment** the download/extract code.
4. Run the notebook top-to-bottom.

The notebook will:

- Download the ZIP file from the Google Drive link provided in the original TennisCourtDetector README.
- Extract the images and annotations into a local folder (e.g. `training/data/`).
- Convert or reorganize the annotations into the JSON format expected by this repository:
  - `training/data/data_train.json`
  - `training/data/data_val.json`
  - and the corresponding `training/data/images/` directory.

Once the notebook finishes:

- You can train the court keypoint model via the script mentioned in the root `README.md` (see the **Keypoint Training** section).
- The same dataset structure will also be used by the court keypoints training notebook itself.

