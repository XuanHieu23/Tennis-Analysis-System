from pathlib import Path

from utils import (read_video, save_video)
from utils import measure_distance, draw_player_stats, convert_pixel_distance_to_meters
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import constants
import cv2
import pandas as pd
from copy import deepcopy

def main():
    # Read video
    input_video_path = "Tennis/input_video.mp4"
    use_cached_detections = True
    video_frames = read_video(input_video_path)
    if not video_frames:
        raise RuntimeError(f"No frames were read from '{input_video_path}'.")

    video_stem = Path(input_video_path).stem
    player_stub_path = f"tracker_stubs/{video_stem}_player_detections.pkl"
    ball_stub_path = f"tracker_stubs/{video_stem}_ball_detections.pkl"

    # Detect players and Ball
    player_tracker = PlayerTracker(model_path="yolo11x.pt")
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=use_cached_detections,
                                                     stub_path=player_stub_path)
    
    ball_tracker = BallTracker(model_path="models/yolo11x_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=use_cached_detections,
                                                 stub_path=ball_stub_path)
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections) # Interpolate missing ball positions
    
    # Court Line Detection
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict_from_frames(video_frames, num_frames=5)
    
    # Choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections,
        ball_detections,
        court_keypoints,
    )

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_number_of_moves': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_number_of_moves': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    last_shooter = None
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        if end_frame <= start_frame:
            continue
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps
        if ball_shot_time_in_seconds <= 0:
            continue

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Player who shot the ball (closest player to ball at shot start frame)
        player_positions = player_mini_court_detections[start_frame]
        if not player_positions:
            continue
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id],
                ball_mini_court_detections[start_frame][1],
            ),
        )
        # Rally hits should typically alternate players; this guards against noisy ball detections
        # repeatedly assigning consecutive hits to the same player.
        if last_shooter is not None and len(player_positions) == 2 and player_shot_ball == last_shooter:
            player_shot_ball = 1 if last_shooter == 2 else 2
        last_shooter = player_shot_ball

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        # Track movement speed for both players in the same rally segment.
        start_positions = player_mini_court_detections[start_frame]
        end_positions = player_mini_court_detections[end_frame]
        for player_id in (1, 2):
            if player_id not in start_positions or player_id not in end_positions:
                continue

            distance_covered_pixels = measure_distance(start_positions[player_id], end_positions[player_id])
            distance_covered_meters = convert_pixel_distance_to_meters(
                distance_covered_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court(),
            )
            player_speed = distance_covered_meters / ball_shot_time_in_seconds * 3.6

            current_player_stats[f'player_{player_id}_number_of_moves'] += 1
            current_player_stats[f'player_{player_id}_total_player_speed'] += player_speed
            current_player_stats[f'player_{player_id}_last_player_speed'] = player_speed

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    shot_denom_p1 = player_stats_data_df['player_1_number_of_shots'].replace(0, pd.NA)
    shot_denom_p2 = player_stats_data_df['player_2_number_of_shots'].replace(0, pd.NA)
    move_denom_p1 = player_stats_data_df['player_1_number_of_moves'].replace(0, pd.NA)
    move_denom_p2 = player_stats_data_df['player_2_number_of_moves'].replace(0, pd.NA)

    player_stats_data_df['player_1_average_shot_speed'] = (
        player_stats_data_df['player_1_total_shot_speed'] / shot_denom_p1
    ).fillna(0)
    player_stats_data_df['player_2_average_shot_speed'] = (
        player_stats_data_df['player_2_total_shot_speed'] / shot_denom_p2
    ).fillna(0)
    player_stats_data_df['player_1_average_player_speed'] = (
        player_stats_data_df['player_1_total_player_speed'] / move_denom_p1
    ).fillna(0)
    player_stats_data_df['player_2_average_player_speed'] = (
        player_stats_data_df['player_2_total_player_speed'] / move_denom_p2
    ).fillna(0)

    # Draw output

    ## Draw Players Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw Balls Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames,
        ball_mini_court_detections,
        color=(0, 255, 255),
    )

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    ## Draw frame number on the top left corner of the video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, f"output_videos/{video_stem}_output.avi")

if __name__ == "__main__":
    main()
