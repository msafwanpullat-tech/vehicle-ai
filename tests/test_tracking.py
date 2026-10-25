import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.services.vehicle_tracker import VehicleTrackerService

def test_bytetrack_tracking_and_counting():
    input_video = "videos/sample_traffic.mp4"
    output_video = "outputs/test_counting_output.mp4"

    print("Initializing VehicleTrackerService with ByteTrack & Line Counting...")
    tracker = VehicleTrackerService(model_path="models/yolo11n.pt")

    print(f"Tracking & Counting vehicles in '{input_video}' -> '{output_video}'...")
    results = tracker.track_video(input_path=input_video, output_path=output_video, line_ratio=0.5)

    print("\n--- PHASE 3 BYTETrack & LINE COUNTING TEST RESULTS ---")
    print(f"Status: {results['status']}")
    print(f"Total Frames Processed: {results['processed_frames']}")
    print(f"Processing Time: {results['processing_time_sec']}s")
    print(f"Inference FPS: {results['processing_fps']}")
    print(f"Total Unique Vehicles Tracked: {results['total_unique_vehicles']}")
    print(f"Total Line Crossings: {results['total_line_crossings']}")
    print(f"Directional Counts - IN: {results['in_count']} | OUT: {results['out_count']}")
    print(f"Class-Wise Line Counts: {results['class_counts']}")

    assert os.path.exists(output_video), "Output video file was not generated!"
    assert results["processed_frames"] > 0, "No frames were processed!"
    print("\n[SUCCESS] ByteTrack & Line Counting test passed successfully!")

if __name__ == "__main__":
    test_bytetrack_tracking_and_counting()
