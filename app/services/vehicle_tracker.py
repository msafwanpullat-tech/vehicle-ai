import os
import time
import logging
import cv2
import numpy as np
from ultralytics import YOLO

from app.services.counter import LineCounter
from app.services.plate_detector import LicensePlateDetector
from app.services.ocr_service import LicensePlateOCRService
from app.services.plate_registry import PlateRegistryService

logger = logging.getLogger("vehicle_ai.vehicle_tracker")
logger.setLevel(logging.INFO)

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CLASS_COLORS = {
    "car": (246, 130, 59),        # Blue/Azure
    "motorcycle": (129, 185, 16), # Emerald Green
    "bus": (11, 158, 245),         # Amber / Orange
    "truck": (68, 68, 239)          # Bright Red
}

class VehicleTrackerService:
    """
    Unified Computer Vision Engine:
    1. YOLO11n Vehicle Detection
    2. ByteTrack Persistent Vehicle Tracking
    3. Virtual Line Crossing Counting
    4. License Plate Detection & OCR Recognition Engine
    """

    def __init__(self, model_path: str = "models/yolo11n.pt", tracker_type: str = "bytetrack.yaml"):
        if not os.path.exists(model_path):
            if os.path.exists("yolo11n.pt"):
                model_path = "yolo11n.pt"
            else:
                raise FileNotFoundError(f"YOLO model weight file missing at '{model_path}'")

        logger.info(f"Initializing YOLO11n vehicle model from '{model_path}'")
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.target_class_ids = list(VEHICLE_CLASSES.keys())

        # License Plate & OCR Engine Services
        self.plate_detector = LicensePlateDetector()
        self.ocr_service = LicensePlateOCRService()
        self.plate_registry = PlateRegistryService()

    def track_video(self, input_path: str, output_path: str, conf_threshold: float = 0.25, line_ratio: float = 0.5) -> dict:
        """
        Runs full video processing pipeline:
        - Vehicle detection & ByteTrack tracking
        - Virtual line crossing vehicle counting
        - License plate detection & OCR recognition
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video stream: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0:
            fps = 30.0
            width = width or 1280
            height = height or 720

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("outputs/plates", exist_ok=True)
        os.makedirs("outputs/frames", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset plate registry for new video session
        self.plate_registry.clear()

        # Initialize virtual line counter
        counter = LineCounter(line_position_ratio=line_ratio)
        counter.configure_frame(width=width, height=height, line_ratio=line_ratio)

        start_time = time.time()
        processed_frames = 0
        total_detections_count = 0

        unique_track_ids_all = set()
        unique_track_ids_by_class = {class_name: set() for class_name in VEHICLE_CLASSES.values()}

        logger.info(f"Starting Phase 4 Pipeline (YOLO + ByteTrack + Counter + Plate OCR) on '{input_path}'")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                processed_frames += 1

                # Calculate video timestamp MM:SS
                sec = processed_frames / fps
                timestamp_str = f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}"

                # Run ByteTrack vehicle tracking
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_type,
                    classes=self.target_class_ids,
                    conf=conf_threshold,
                    verbose=False
                )[0]

                boxes = results.boxes
                active_frame_tracks = 0

                if boxes is not None and len(boxes) > 0:
                    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                    for idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Clip coordinates to frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        class_name = VEHICLE_CLASSES.get(cls_id, "vehicle")

                        total_detections_count += 1
                        color = CLASS_COLORS.get(class_name, (0, 255, 0))

                        # Center point calculation for line crossing
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        reg_label_part = ""

                        if track_ids is not None and idx < len(track_ids):
                            track_id = int(track_ids[idx])
                            active_frame_tracks += 1
                            unique_track_ids_all.add(track_id)
                            unique_track_ids_by_class[class_name].add(track_id)

                            # --- LICENSE PLATE DETECTION & OCR PIPELINE ---
                            if self.plate_registry.should_run_ocr(track_id, processed_frames, ocr_interval=15):
                                vehicle_roi = frame[y1:y2, x1:x2]
                                if vehicle_roi.size > 0:
                                    plate_crop, plate_bbox = self.plate_detector.detect_plate(vehicle_roi)
                                    if plate_crop is not None and plate_crop.size > 0:
                                        plate_num, ocr_conf = self.ocr_service.recognize_plate(plate_crop)

                                        # Save snapshot URLs if plate recognized or candidate found
                                        plate_crop_filename = f"plate_id{track_id}.jpg"
                                        frame_filename = f"frame_{processed_frames}_id{track_id}.jpg"

                                        crop_disk_path = os.path.join("outputs/plates", plate_crop_filename)
                                        frame_disk_path = os.path.join("outputs/frames", frame_filename)

                                        cv2.imwrite(crop_disk_path, plate_crop)
                                        cv2.imwrite(frame_disk_path, frame)

                                        crop_url = f"/outputs/plates/{plate_crop_filename}"
                                        frame_url = f"/outputs/frames/{frame_filename}"

                                        # Update plate registry record
                                        self.plate_registry.add_or_update_record(
                                            track_id=track_id,
                                            vehicle_type=class_name,
                                            registration_number=plate_num,
                                            confidence=ocr_conf,
                                            frame_number=processed_frames,
                                            timestamp=timestamp_str,
                                            frame_url=frame_url,
                                            crop_url=crop_url
                                        )

                            # Fetch current plate record from registry
                            record = self.plate_registry.records.get(track_id)
                            if record and record.get("registration_number") != "Unknown":
                                plate_text = record["registration_number"]
                                plate_conf = record["confidence"]
                                reg_label_part = f" | {plate_text} ({plate_conf:.0f}%)"

                            label = f"{class_name.upper()} | ID: {track_id}{reg_label_part}"

                            # Update Line Counter
                            counter.update_track(track_id, class_name, (cx, cy))
                        else:
                            label = f"{class_name.upper()} {conf:.2f}"

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw center dot
                        cv2.circle(frame, (cx, cy), 4, color, -1)

                        # Draw text background badge
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                        badge_y1 = max(0, y1 - text_h - 10)
                        cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 10, badge_y1 + text_h + 8), color, -1)

                        # Draw label text
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 5, badge_y1 + text_h + 3),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA
                        )

                # --- DRAW VIRTUAL COUNTING LINE ---
                line_color = (0, 255, 255)
                cv2.line(frame, (0, counter.line_y), (width, counter.line_y), line_color, 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"--- VIRTUAL COUNTING LINE (Y={int(counter.line_ratio * 100)}%) ---",
                    (width - 450, counter.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    line_color,
                    1,
                    cv2.LINE_AA
                )

                # --- DRAW OVERLAY HUD PANEL ---
                hud_h = 75
                hud_w = 420
                cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (15, 23, 42), -1)
                cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (51, 65, 85), 1)

                line_summary = counter.get_summary()

                hud_line1 = f"LINE CROSSED: {line_summary['total_line_crossings']} | IN: {line_summary['in_count']} | OUT: {line_summary['out_count']}"
                cv2.putText(frame, hud_line1, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

                lc = line_summary['class_line_counts']
                hud_line2 = f"CAR: {lc.get('car', 0)} | M.CYCLE: {lc.get('motorcycle', 0)} | BUS: {lc.get('bus', 0)} | TRUCK: {lc.get('truck', 0)}"
                cv2.putText(frame, hud_line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

                out.write(frame)

        finally:
            cap.release()
            out.release()

        elapsed_time = time.time() - start_time
        processing_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

        logger.info(f"Phase 4 processing completed in {elapsed_time:.2f}s ({processing_fps:.1f} FPS)")

        line_stats = counter.get_summary()
        recognized_plates = self.plate_registry.get_all_records()

        return {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "processing_time_sec": round(elapsed_time, 2),
            "processing_fps": round(processing_fps, 2),
            "original_fps": round(fps, 2),
            "resolution": f"{width}x{height}",
            "total_unique_vehicles": len(unique_track_ids_all),
            "total_line_crossings": line_stats["total_line_crossings"],
            "in_count": line_stats["in_count"],
            "out_count": line_stats["out_count"],
            "class_counts": line_stats["class_line_counts"],
            "unique_tracked_class_counts": {
                class_name: len(track_set) for class_name, track_set in unique_track_ids_by_class.items()
            },
            "total_vehicle_detections": total_detections_count,
            "line_position_ratio": line_ratio,
            "recognized_plates": recognized_plates
        }
