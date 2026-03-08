from utils import (read_video, save_video)

def main():
    input_video_path = "Tennis/input_video.mp4"
    video_frames = read_video(input_video_path)

    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()