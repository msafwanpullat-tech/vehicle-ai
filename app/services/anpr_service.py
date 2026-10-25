import json
import logging
import math
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.services.ocr_service import LicensePlateOCRService
from app.services.plate_detector import LicensePlateDetector
from app.services.registration_search import (
    INDIAN_REGISTRATION_PATTERN,
    format_video_timestamp,
    normalize_registration,
)

logger = logging.getLogger("vehicle_ai.anpr")


class OCRService(Protocol):
    def recognize_plate(self, plate_crop: np.ndarray) -> tuple[str, float]: ...

    @staticmethod
    def normalize_registration(raw_text: str) -> str: ...


class PlateRecognitionProcessor:
    def __init__(
        self,
        detector: LicensePlateDetector | None = None,
        ocr_service: OCRService | None = None,
        ocr_interval: int = 15,
        minimum_confidence: float = 70.0,
        debug_dir: Path | None = None,
        analysis_dir: Path | None = None,
        session_dir: Path | None = None,
    ) -> None:
        self.detector = detector or LicensePlateDetector()
        self.ocr_service = ocr_service or LicensePlateOCRService()
        self.ocr_interval = ocr_interval
        self.minimum_confidence = minimum_confidence
        self.debug_dir = debug_dir
        self.analysis_dir = analysis_dir
        self.session_dir = session_dir
        self.video_fps: float | None = None
        self.detections: list[dict[str, object]] = []
        self.tracking_events: list[dict[str, object]] = []
        self.search_results: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.last_scan_frame: dict[int, int] = {}
        self.best_crop_quality: dict[int, float] = {}
        self.records: dict[int, dict[str, int | float | str]] = {}
        self.ocr_attempts = 0
        self.ocr_available: bool | None = None

    def set_video_fps(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")
        self.video_fps = fps

    def save_analysis_manifest(
        self, output_video_url: str, source_video_url: str
    ) -> None:
        if self.session_dir is None:
            raise RuntimeError("ANPR analysis storage was not initialized.")
        manifest = self.build_manifest(output_video_url, source_video_url)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.session_dir / "analysis.json"
        temporary_path = self.session_dir / "analysis.json.tmp"
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        temporary_path.replace(manifest_path)

    def build_manifest(
        self, output_video_url: str, source_video_url: str
    ) -> dict[str, object]:
        return {
            **self.summary,
            "fps": self.video_fps,
            "output_video_url": output_video_url,
            "source_video_url": source_video_url,
            "detections": self.detections,
            "tracking_events": self.tracking_events,
            "search_results": self.search_results,
        }

    def add_tracking_event(
        self,
        frame_number: int,
        track_id: int,
        vehicle_class: str,
        line_crossing: bool,
        crossing_direction: str | None,
        counted_for_total: bool = False,
    ) -> None:
        if self.video_fps is None:
            logger.warning("Video FPS was not set before storing tracking events; defaulting to 30.0.")
            self.video_fps = 30.0
        seconds, timestamp = format_video_timestamp(frame_number, self.video_fps)
        self.tracking_events.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": seconds,
                "timestamp": timestamp,
                "vehicle_id": track_id,
                "vehicle_class": vehicle_class,
                "line_crossing": line_crossing,
                "crossing_direction": crossing_direction or "",
                "counted_for_total": counted_for_total,
            }
        )

    def save_search_result(self, result: dict[str, object]) -> None:
        self.search_results.append(result)

    def process_vehicle(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
        track_id: int,
        vehicle_class: str,
        frame_number: int,
    ) -> tuple[dict[str, int | float | str] | None, tuple[int, int, int, int] | None]:
        previous_scan = self.last_scan_frame.get(track_id)
        if previous_scan is not None and frame_number - previous_scan < self.ocr_interval:
            return None, None
        self.last_scan_frame[track_id] = frame_number

        x1, y1, x2, y2 = vehicle_bbox
        vehicle_roi = frame[y1:y2, x1:x2]
        if vehicle_roi.size == 0:
            return None, None

        detect_candidates = getattr(self.detector, "detect_candidates", None)
        if callable(detect_candidates):
            candidates = detect_candidates(vehicle_roi)
        else:
            crop, bbox = self.detector.detect_plate(vehicle_roi)
            candidates = [(crop, bbox)] if crop is not None and bbox is not None else []

        valid_candidates: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
        for crop, bbox in candidates:
            quality = self._crop_quality(crop)
            if quality is not None:
                valid_candidates.append((crop, bbox, quality))

        if not valid_candidates:
            return None, None

        valid_candidates.sort(key=lambda item: item[2], reverse=True)

        best_frame_quality = valid_candidates[0][2]
        prev_quality = self.best_crop_quality.get(track_id, -1.0)
        if best_frame_quality <= prev_quality:
            return None, None

        selected_candidate = None
        for crop, bbox, quality in valid_candidates[:3]:
            self.ocr_attempts += 1
            try:
                raw_text, confidence = self.ocr_service.recognize_plate(crop)
                self.ocr_available = True
            except Exception:
                logger.exception("Local OCR failed for a detected plate candidate.")
                self.ocr_available = False
                record = self._build_record(
                    track_id, vehicle_class, "OCR unavailable", 0.0, "OCR unavailable"
                )
                self._save_analysis_detection(
                    vehicle_roi,
                    crop,
                    track_id,
                    vehicle_class,
                    frame_number,
                    "",
                    0.0,
                    "OCR unavailable",
                )
                self._save_debug_artifacts(
                    frame, vehicle_roi, crop, bbox, vehicle_bbox,
                    track_id, vehicle_class, frame_number, record, [],
                )
                return self._save_record(record), bbox

            registration = self.ocr_service.normalize_registration(raw_text)
            is_registration = bool(INDIAN_REGISTRATION_PATTERN.fullmatch(registration))

            cand_info = {
                "crop": crop,
                "bbox": bbox,
                "quality": quality,
                "raw_text": raw_text,
                "registration": registration,
                "is_registration": is_registration,
                "confidence": confidence,
            }

            if is_registration and confidence >= self.minimum_confidence:
                selected_candidate = cand_info
                break

            if selected_candidate is None:
                selected_candidate = cand_info
            elif is_registration and not selected_candidate["is_registration"]:
                selected_candidate = cand_info
            elif is_registration and selected_candidate["is_registration"] and confidence > selected_candidate["confidence"]:
                selected_candidate = cand_info
            elif not selected_candidate["is_registration"]:
                if len(registration) >= 4 and len(selected_candidate["registration"]) < 4:
                    selected_candidate = cand_info
                elif len(registration) >= 4 and confidence > selected_candidate["confidence"]:
                    selected_candidate = cand_info

        if selected_candidate is None:
            return None, None

        plate_crop = selected_candidate["crop"]
        plate_bbox = selected_candidate["bbox"]
        quality = selected_candidate["quality"]
        raw_text = selected_candidate["raw_text"]
        registration = selected_candidate["registration"]
        is_registration = selected_candidate["is_registration"]
        confidence = selected_candidate["confidence"]

        self.best_crop_quality[track_id] = max(prev_quality, quality)

        if is_registration and confidence >= self.minimum_confidence:
            status = "Recognized"
            plate_text = registration
        elif len(registration) >= 4:
            status = "Uncertain"
            plate_text = "Uncertain"
        else:
            status = "Unreadable"
            plate_text = "Unreadable"

        record = self._build_record(
            track_id, vehicle_class, plate_text, confidence, status, candidate_text=registration
        )
        self._save_analysis_detection(
            vehicle_roi,
            plate_crop,
            track_id,
            vehicle_class,
            frame_number,
            raw_text,
            confidence,
            status,
        )
        self._save_debug_artifacts(
            frame,
            vehicle_roi,
            plate_crop,
            plate_bbox,
            vehicle_bbox,
            track_id,
            vehicle_class,
            frame_number,
            record,
            getattr(self.ocr_service, "last_debug_results", []),
        )
        return self._save_record(record), plate_bbox

    @staticmethod
    def _crop_quality(plate_crop: np.ndarray) -> float | None:
        if plate_crop is None or plate_crop.size == 0:
            return None
        height, width = plate_crop.shape[:2]
        if width < 60 or height < 18:
            return None
        aspect_ratio = width / height
        if not 1.2 <= aspect_ratio <= 10.0:
            return None

        geometry_score = max(0.1, 1.0 - abs(aspect_ratio - 3.0) / 5.0)
        resolution_score = min(1.0, width / 100.0) * min(1.0, height / 25.0)
        if plate_crop.ndim == 2:
            gray = plate_crop
        else:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not math.isfinite(sharpness):
            return None
        sharpness_score = sharpness / (sharpness + 100.0)
        return geometry_score * resolution_score * sharpness_score

    def _save_analysis_detection(
        self,
        vehicle_roi: np.ndarray,
        plate_crop: np.ndarray,
        track_id: int,
        vehicle_class: str,
        frame_number: int,
        ocr_text: str,
        confidence: float,
        status: str,
    ) -> None:
        if self.analysis_dir is None:
            return
        if self.video_fps is None:
            logger.warning("Video FPS was not set before storing ANPR detections; defaulting to 30.0.")
            self.video_fps = 30.0
        seconds, timestamp = format_video_timestamp(frame_number, self.video_fps)
        stem = f"frame_{frame_number:06d}_track_{track_id}"
        vehicle_filename = f"{stem}_vehicle.jpg"
        plate_filename = f"{stem}_plate.jpg"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        for filename, image in (
            (vehicle_filename, vehicle_roi),
            (plate_filename, plate_crop),
        ):
            if not cv2.imwrite(str(self.analysis_dir / filename), image):
                raise OSError(f"Could not save ANPR screenshot {filename}.")

        debug_results = getattr(self.ocr_service, "last_debug_results", [])
        ocr_candidates = [
            {
                "text": str(result.get("normalized_text", "")),
                "confidence": float(result.get("confidence", 0)),
            }
            for result in debug_results
            if result.get("normalized_text")
        ]
        if ocr_text and not any(
            candidate["text"] == normalize_registration(ocr_text)
            for candidate in ocr_candidates
        ):
            ocr_candidates.append(
                {"text": normalize_registration(ocr_text), "confidence": confidence}
            )

        analysis_name = self.analysis_dir.name
        self.detections.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": seconds,
                "timestamp": timestamp,
                "vehicle_id": track_id,
                "vehicle_class": vehicle_class,
                "ocr_text": normalize_registration(ocr_text),
                "plate_text": (
                    normalize_registration(ocr_text) if status == "Recognized" else status
                ),
                "candidate_text": normalize_registration(ocr_text),
                "ocr_confidence": round(confidence, 1),
                "status": status,
                "ocr_candidates": ocr_candidates,
                "vehicle_image_url": (
                    f"/outputs/counting/{analysis_name}/{vehicle_filename}"
                ),
                "plate_image_url": f"/outputs/counting/{analysis_name}/{plate_filename}",
            }
        )

    def _save_debug_artifacts(
        self,
        frame: np.ndarray,
        vehicle_roi: np.ndarray,
        plate_crop: np.ndarray,
        plate_bbox: tuple[int, int, int, int],
        vehicle_bbox: tuple[int, int, int, int],
        track_id: int,
        vehicle_class: str,
        frame_number: int,
        record: dict[str, int | float | str],
        ocr_results: list[dict[str, str | float]],
    ) -> None:
        if self.debug_dir is None:
            return
        stem = f"frame_{frame_number:06d}_track_{track_id}"
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = vehicle_bbox
            px1, py1, px2, py2 = plate_bbox
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.rectangle(
                annotated,
                (x1 + px1, y1 + py1),
                (x1 + px2, y1 + py2),
                (255, 180, 40),
                2,
            )
            images = [
                (f"{stem}_frame.jpg", annotated),
                (f"{stem}_vehicle.jpg", vehicle_roi),
                (f"{stem}_candidate.jpg", plate_crop),
            ]
            if isinstance(self.ocr_service, LicensePlateOCRService):
                for variant_name, variant in self.ocr_service.prepare_plate_variants(plate_crop):
                    images.append((f"{stem}_{variant_name}.jpg", variant))
            for filename, image in images:
                if not cv2.imwrite(str(self.debug_dir / filename), image):
                    raise OSError(f"OpenCV could not write {filename}")
            details = {
                "frame_number": frame_number,
                "vehicle_id": track_id,
                "vehicle_class": vehicle_class,
                "vehicle_bbox": vehicle_bbox,
                "candidate_bbox_in_vehicle": plate_bbox,
                "candidate_shape": list(plate_crop.shape),
                "ocr_results": ocr_results,
                "selected_result": record,
            }
            with (self.debug_dir / f"{stem}.json").open("w", encoding="utf-8") as file:
                json.dump(details, file, ensure_ascii=True, indent=2)
        except OSError:
            logger.exception("Could not save optional ANPR debug artifacts.")

    def _build_record(
        self,
        track_id: int,
        vehicle_class: str,
        plate_text: str,
        confidence: float,
        status: str,
        candidate_text: str = "",
    ) -> dict[str, int | float | str]:
        cand = candidate_text or (plate_text if plate_text not in ("Uncertain", "Unreadable", "OCR unavailable") else "")
        return {
            "vehicle_id": track_id,
            "vehicle_class": vehicle_class,
            "plate_text": plate_text,
            "candidate_text": cand,
            "ocr_confidence": round(max(0.0, min(100.0, confidence)), 1),
            "status": status,
        }

    def _save_record(
        self, record: dict[str, int | float | str]
    ) -> dict[str, int | float | str]:
        vehicle_id = int(record["vehicle_id"])
        previous = self.records.get(vehicle_id)
        if previous is None:
            self.records[vehicle_id] = record
        elif previous["status"] != "Recognized" and (
            record["status"] == "Recognized"
            or float(record["ocr_confidence"]) > float(previous["ocr_confidence"])
        ):
            self.records[vehicle_id] = record
        return self.records[vehicle_id]

    def get_records(self) -> list[dict[str, int | float | str]]:
        return [self.records[key] for key in sorted(self.records)]

    @property
    def status(self) -> str:
        if self.ocr_available is False:
            return "unavailable"
        if self.ocr_available is True:
            return "processed"
        return "no_candidates"
