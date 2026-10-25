import os
from app.services.video_processor import VehicleDetectorService

def test_video_detection():
    input_video = "videos/sample_traffic.mp4"
    output_video = "outputs/test_output.mp4"

    print("Initializing VehicleDetectorService...")
    service = VehicleDetectorService(model_path="models/yolo11n.pt")

    print(f"Processing '{input_video}' -> '{output_video}'...")
    results = service.process_video(input_path=input_video, output_path=output_video, conf_threshold=0.25)

    print("\n--- TEST RESULTS ---")
    print(f"Status: {results['status']}")
    print(f"Total Frames: {results['total_frames']}")
    print(f"Processed Frames: {results['processed_frames']}")
    print(f"Processing Time: {results['processing_time_sec']}s")
    print(f"Inference FPS: {results['processing_fps']}")
    print(f"Total Vehicle Detections: {results['total_vehicle_detections']}")
    print(f"Class Counts: {results['class_counts']}")
    
    assert os.path.exists(output_video), "Output video file was not created!"
    assert results["total_frames"] == 60, f"Expected 60 frames, got {results['total_frames']}"
    assert results["total_vehicle_detections"] > 0, "Expected at least 1 vehicle detection!"
    print("\n✅ All Vehicle Detection tests passed successfully!")

if __name__ == "__main__":
    test_video_detection()
