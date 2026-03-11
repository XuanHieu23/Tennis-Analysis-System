from ultralytics import YOLO

PLAYER_MODEL_PATH = "yolo11x.pt"
BALL_MODEL_PATH = "models/yolo11x_best.pt"
SOURCE_VIDEO = "Tennis/input_video.mp4"

player_model = YOLO(PLAYER_MODEL_PATH)
ball_model = YOLO(BALL_MODEL_PATH)

player_results = player_model.track(
    SOURCE_VIDEO,
    classes=[0],
    conf=0.15,
    tracker="bytetrack.yaml",
    save=True,
    verbose=False,
)
ball_results = ball_model.predict(
    SOURCE_VIDEO,
    conf=0.10,
    save=True,
    verbose=False,
)

print(f"Player track frames: {len(player_results)}")
print(f"Ball detect frames: {len(ball_results)}")
