import logging
from app.services.vehicle_tracker import VehicleTrackerService

logger = logging.getLogger("vehicle_ai.video_processor")

class VehicleDetectorService:
    """
    Video processor wrapper utilizing VehicleTrackerService with ByteTrack.
    Provides backward compatibility for existing API endpoints while supporting multi-object vehicle tracking.
    """

    def __init__(self, model_path: str = "models/yolo11n.pt"):
        self.tracker = VehicleTrackerService(model_path=model_path, tracker_type="bytetrack.yaml")

    def process_video(self, input_path: str, output_path: str, conf_threshold: float = 0.25) -> dict:
        """Delegates video processing to VehicleTrackerService with ByteTrack tracking."""
        return self.tracker.track_video(input_path=input_path, output_path=output_path, conf_threshold=conf_threshold)
