from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    # Read video
    input_video_path = "Tennis/input_video2.mp4"
    video_frames = read_video(input_video_path)

    # Detect players and Ball
    player_tracker = PlayerTracker(model_path="yolo11x.pt")
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stubs=False,
                                                     stubs_path="tracker_stubs/player_detections.pkl")
    
    ball_tracker = BallTracker(model_path="models/yolo11x_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stubs=False,
                                                 stubs_path="tracker_stubs/ball_detections.pkl")
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections) # Interpolate missing ball positions
    
    # Court Line Detection
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0]) # Predict court lines on the first frame
    
    # Choose players

    # Draw output

    ## Draw Players Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw Balls Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw frame number on the top left corner of the video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video2.avi")

if __name__ == "__main__":
    main()