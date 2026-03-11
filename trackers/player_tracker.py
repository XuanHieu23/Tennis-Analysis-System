import os
import pickle

import cv2
from ultralytics import YOLO

from utils import measure_distance
from utils import get_foot_position

class PlayerTracker:
    def __init__(self, model_path, conf=0.15, iou=0.5, tracker_cfg="bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.tracker_cfg = tracker_cfg

    def choose_and_filter_players(self, court_keypoints, player_detections):
        filtered_player_detections = []
        previous_player_positions = {1: None, 2: None}

        for frame_player_dict in player_detections:
            filtered_player_dict = self._select_players_for_frame(
                court_keypoints,
                frame_player_dict,
                previous_player_positions,
            )
            for logical_player_id, bbox in filtered_player_dict.items():
                previous_player_positions[logical_player_id] = get_foot_position(bbox)
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        if not player_dict:
            return []

        x_coords = court_keypoints[::2]
        y_coords = court_keypoints[1::2]
        court_mid_x = (min(x_coords) + max(x_coords)) / 2.0
        court_mid_y = (min(y_coords) + max(y_coords)) / 2.0
        top_target = (court_mid_x, min(y_coords))
        bottom_target = (court_mid_x, max(y_coords))

        candidates = []
        for track_id, bbox in player_dict.items():
            foot_pos = get_foot_position(bbox)
            if foot_pos[1] <= court_mid_y:
                dist = measure_distance(foot_pos, top_target)
            else:
                dist = measure_distance(foot_pos, bottom_target)
            candidates.append((track_id, dist))

        candidates.sort(key=lambda x: x[1])
        return [item[0] for item in candidates[:2]]

    def _select_players_for_frame(self, court_keypoints, frame_player_dict, previous_player_positions):
        if not frame_player_dict:
            return {}

        x_coords = court_keypoints[::2]
        y_coords = court_keypoints[1::2]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        court_mid_y = (min_y + max_y) / 2.0

        x_pad = (max_x - min_x) * 0.08
        y_pad = (max_y - min_y) * 0.15
        valid_y_max = max_y + y_pad * 1.8

        candidates = []
        for track_id, bbox in frame_player_dict.items():
            foot_pos = get_foot_position(bbox)
            if (min_x - x_pad) <= foot_pos[0] <= (max_x + x_pad) and (min_y - y_pad) <= foot_pos[1] <= valid_y_max:
                candidates.append((track_id, bbox, foot_pos))

        # Fallback to all person detections if the geometric filter is too strict.
        if len(candidates) < 2:
            candidates = [(track_id, bbox, get_foot_position(bbox)) for track_id, bbox in frame_player_dict.items()]

        if not candidates:
            return {}

        top_candidates = [c for c in candidates if c[2][1] <= court_mid_y]
        bottom_candidates = [c for c in candidates if c[2][1] > court_mid_y]

        if not top_candidates:
            top_candidates = sorted(candidates, key=lambda c: c[2][1])[:2]
        if not bottom_candidates:
            bottom_candidates = sorted(candidates, key=lambda c: c[2][1], reverse=True)[:2]

        top_target = ((min_x + max_x) / 2.0, min_y)
        bottom_target = ((min_x + max_x) / 2.0, max_y)

        selected = {}
        used_track_ids = set()

        top_pick = self._pick_candidate(
            top_candidates,
            used_track_ids,
            previous_player_positions[2],
            top_target,
        )
        if top_pick is None:
            top_pick = self._pick_candidate(candidates, used_track_ids, previous_player_positions[2], top_target)
        if top_pick is not None:
            used_track_ids.add(top_pick[0])
            selected[2] = top_pick[1]

        bottom_pick = self._pick_candidate(
            bottom_candidates,
            used_track_ids,
            previous_player_positions[1],
            bottom_target,
        )
        if bottom_pick is None:
            bottom_pick = self._pick_candidate(candidates, used_track_ids, previous_player_positions[1], bottom_target)
        if bottom_pick is not None:
            used_track_ids.add(bottom_pick[0])
            selected[1] = bottom_pick[1]

        if len(selected) == 1:
            # If only one side has a detection, keep at least one player ID present for downstream logic.
            _, only_bbox, only_foot = next(iter(candidates))
            fallback_id = 1 if only_foot[1] > court_mid_y else 2
            selected.setdefault(fallback_id, only_bbox)

        return selected

    def _pick_candidate(self, candidates, used_track_ids, previous_position, target_position):
        available = [candidate for candidate in candidates if candidate[0] not in used_track_ids]
        if not available:
            return None

        if previous_position is not None:
            return min(available, key=lambda candidate: measure_distance(candidate[2], previous_position))
        return min(available, key=lambda candidate: measure_distance(candidate[2], target_position))

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            if len(player_detections) == len(frames):
                return player_detections
            print(
                f"[PlayerTracker] Ignoring stub '{stub_path}' because it has {len(player_detections)} "
                f"frames while input has {len(frames)} frames."
            )
            player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_cfg,
            verbose=False,
        )[0]
        id_name_dict = results.names

        player_dict = {}
        for box_index, box in enumerate(results.boxes):
            track_id = int(box.id.tolist()[0]) if box.id is not None else -(box_index + 1)
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
