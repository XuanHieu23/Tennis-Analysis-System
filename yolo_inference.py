from ultralytics import YOLO

# detect_model = YOLO('training/runs/detect/train3/weights/last.pt')
track_model = YOLO('yolo12x')

# result = detect_model.predict('Tennis/input_video.mp4', save=True)
result = track_model.track('Tennis/input_video.mp4', conf=0.2, save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)