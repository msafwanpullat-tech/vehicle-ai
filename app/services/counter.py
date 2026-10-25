import logging
from typing import Dict, Set, Tuple, Optional

logger = logging.getLogger("vehicle_ai.counter")

class LineCounter:
    """
    Virtual Line Crossing Counter for vehicles.
    Tracks vehicle center trajectory across frames and counts unique tracking IDs
    when they cross a user-configurable line segment.
    """

    def __init__(self, line_position_ratio: float = 0.5):
        """
        :param line_position_ratio: Normalized Y position of counting line (0.1 to 0.9 of frame height)
        """
        self.line_ratio = max(0.1, min(0.9, line_position_ratio))
        self.line_y: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

        # Persistent track history: track_id -> list of (cx, cy)
        self.track_history: Dict[int, list] = {}

        # Set of already counted tracking IDs to prevent duplicate counting
        self.counted_ids: Set[int] = set()

        # Class-wise & Directional Counters
        self.class_counts: Dict[str, int] = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }
        self.in_count: int = 0
        self.out_count: int = 0
        self.total_count: int = 0

    def configure_frame(self, width: int, height: int, line_ratio: Optional[float] = None):
        """Configures counting line geometry according to video dimensions."""
        if line_ratio is not None:
            self.line_ratio = max(0.1, min(0.9, line_ratio))
            
        self.frame_width = width
        self.frame_height = height
        self.line_y = int(height * self.line_ratio)
        logger.info(f"Configured Virtual Counting Line at Y={self.line_y} ({int(self.line_ratio * 100)}% height)")

    def update_track(self, track_id: int, class_name: str, center: Tuple[int, int]) -> Optional[Dict]:
        """
        Updates trajectory history for track_id and checks for line crossing.
        
        :param track_id: ByteTrack vehicle tracking ID
        :param class_name: Vehicle class name ('car', 'motorcycle', 'bus', 'truck')
        :param center: (cx, cy) center coordinate of vehicle bounding box
        :return: Dict with crossing event details if line was crossed, else None
        """
        cx, cy = center
        crossing_event = None

        if track_id in self.track_history and len(self.track_history[track_id]) > 0:
            prev_cx, prev_cy = self.track_history[track_id][-1]

            # Check if vehicle crossed the line between prev frame and current frame
            # Condition: prev_cy and cy lie on opposite sides of line_y
            if (prev_cy < self.line_y <= cy) or (prev_cy > self.line_y >= cy):
                # Calculate exact X intersection point on line_y
                if cy != prev_cy:
                    x_intersect = prev_cx + (self.line_y - prev_cy) * (cx - prev_cx) / (cy - prev_cy)
                else:
                    x_intersect = prev_cx

                # Ensure intersection point falls within video frame bounds
                if 0 <= x_intersect <= self.frame_width:
                    # Prevent duplicate counting of the same tracking ID
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        self.total_count += 1

                        if class_name in self.class_counts:
                            self.class_counts[class_name] += 1
                        else:
                            self.class_counts[class_name] = 1

                        # Determine direction:
                        # Moving downwards (prev_cy < line_y <= cy) -> IN
                        # Moving upwards (prev_cy > line_y >= cy) -> OUT
                        direction = "IN" if prev_cy < self.line_y else "OUT"
                        if direction == "IN":
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        crossing_event = {
                            "track_id": track_id,
                            "class_name": class_name,
                            "direction": direction,
                            "total_count": self.total_count
                        }
                        logger.info(f"Vehicle {class_name.upper()} [ID: {track_id}] CROSSED LINE ({direction}). Total Count: {self.total_count}")

        # Update history buffer (keep last 30 center points for smooth trajectory)
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        self.track_history[track_id].append((cx, cy))
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)

        return crossing_event

    def get_summary(self) -> dict:
        """Returns statistics summary dictionary for line crossing counts."""
        return {
            "total_line_crossings": self.total_count,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "class_line_counts": dict(self.class_counts),
            "line_position_y": self.line_y,
            "line_position_ratio": self.line_ratio
        }
