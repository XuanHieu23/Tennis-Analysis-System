import pickle
import os

import cv2
import pandas as pd
from ultralytics import YOLO

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.last_ball_center = None

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill() # Lấp giá trị rỗng

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        # Lower threshold to detect shorter rally segments in broadcast clips.
        minimum_change_frames_for_hit = 12
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        # Bộ lọc chống nhiễu: Xóa bỏ các cú đánh bị trùng lặp sát nhau
        filtered_hits = []
        for frame in frame_nums_with_ball_hits:
            if len(filtered_hits) == 0 or frame - filtered_hits[-1] > 20: 
                filtered_hits.append(frame)
                
        return filtered_hits

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            if len(ball_detections) == len(frames):
                return ball_detections
            print(
                f"[BallTracker] Ignoring stub '{stub_path}' because it has {len(ball_detections)} "
                f"frames while input has {len(frames)} frames."
            )
            ball_detections = []

        # Reset temporal state for each new video sequence.
        self.last_ball_center = None
        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15, verbose=False)[0]

        ball_dict = {}
        if len(results.boxes) == 0:
            return ball_dict

        best_conf = 0
        best_bbox = None

        # Lấy kích thước video để tạo vùng cấm linh hoạt
        height, width = frame.shape[:2]

        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            conf = float(box.conf.item())
            
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            # Bộ lọc kích thước: Bóng không bao giờ to quá 40 pixel
            if w > 40 or h > 40:
                continue

            # Bộ lọc không gian:
            # Chặn hoàn toàn 1/3 màn hình ở góc trên bên trái
            if center_x < (width / 3) and center_y < (height / 3):
                continue
                
            # Chọn vật thể có độ tự tin cao nhất sau khi đã qua màng lọc
            if conf > best_conf:
                best_conf = conf
                best_bbox = bbox

        if best_bbox is not None:
            ball_dict[1] = best_bbox
            
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    
