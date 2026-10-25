import os
import cv2
import numpy as np

def generate_moving_traffic_video(output_video_path: str = "videos/sample_traffic_moving.mp4", num_frames: int = 60, fps: int = 30):
    """
    Generates a sample traffic video with vehicles physically moving down the frame crossing Y=50% line.
    """
    if not os.path.exists("bus.jpg"):
        raise FileNotFoundError("Source image 'bus.jpg' not found.")

    base_img = cv2.imread("bus.jpg")
    h, w, _ = base_img.shape
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for i in range(num_frames):
        # Vertical shift downward across frames
        y_shift = int(i * 8)
        M = np.float32([[1, 0, 0], [0, 1, y_shift - 150]])
        frame = cv2.warpAffine(base_img, M, (w, h))
        out.write(frame)

    out.release()
    print(f"Sample moving traffic video generated at: {output_video_path} ({num_frames} frames)")

if __name__ == "__main__":
    generate_moving_traffic_video()
