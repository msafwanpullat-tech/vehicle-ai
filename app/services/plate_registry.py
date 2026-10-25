import os
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("vehicle_ai.plate_registry")

class PlateRegistryService:
    """
    In-memory / Persistent storage manager for vehicle-to-plate associations,
    OCR periodic throttling, and registration-number search.
    """

    def __init__(self):
        # vehicle_records: track_id -> dict
        self.records: Dict[int, dict] = {}
        # last_ocr_frame: track_id -> frame_number
        self.last_ocr_frame: Dict[int, int] = {}

    def clear(self):
        """Clears all stored plate records for a new video processing session."""
        self.records.clear()
        self.last_ocr_frame.clear()

    def should_run_ocr(self, track_id: int, current_frame: int, ocr_interval: int = 15) -> bool:
        """
        Determines whether OCR should be run for a tracked vehicle ID.
        Throttles OCR requests to avoid repeating OCR on every single frame.
        """
        if track_id not in self.records:
            return True

        record = self.records[track_id]

        # If a high confidence registration number (>= 80%) is already saved, skip repeated OCR
        if record.get("confidence", 0.0) >= 80.0 and record.get("registration_number") != "Unknown":
            return False

        # Otherwise, run OCR periodically every `ocr_interval` frames
        last_frame = self.last_ocr_frame.get(track_id, 0)
        return (current_frame - last_frame) >= ocr_interval

    def add_or_update_record(
        self,
        track_id: int,
        vehicle_type: str,
        registration_number: str,
        confidence: float,
        frame_number: int,
        timestamp: str,
        frame_url: Optional[str] = None,
        crop_url: Optional[str] = None
    ):
        """
        Stores or updates the best recognized plate result for a given vehicle ID.
        """
        self.last_ocr_frame[track_id] = frame_number

        # If vehicle ID doesn't exist yet in registry, create entry
        if track_id not in self.records:
            self.records[track_id] = {
                "track_id": track_id,
                "vehicle_type": vehicle_type,
                "registration_number": registration_number,
                "confidence": round(confidence, 1),
                "frame_number": frame_number,
                "timestamp": timestamp,
                "frame_url": frame_url,
                "crop_url": crop_url
            }
            logger.info(f"Registered Plate [ID {track_id}]: {registration_number} ({confidence}%) @ {timestamp}")
            return

        # If entry exists, update ONLY if new OCR result has higher confidence or replaces 'Unknown'
        existing = self.records[track_id]
        if (registration_number != "Unknown" and existing["registration_number"] == "Unknown") or confidence > existing.get("confidence", 0.0):
            existing["registration_number"] = registration_number
            existing["confidence"] = round(confidence, 1)
            existing["frame_number"] = frame_number
            existing["timestamp"] = timestamp
            if frame_url:
                existing["frame_url"] = frame_url
            if crop_url:
                existing["crop_url"] = crop_url
            logger.info(f"Updated Plate [ID {track_id}]: {registration_number} ({confidence}%) @ {timestamp}")

    def get_all_records(self) -> List[dict]:
        """Returns list of all recognized vehicle plate records sorted by track ID."""
        return sorted(list(self.records.values()), key=lambda x: x["track_id"])

    def search_registration(self, query: str) -> List[dict]:
        """
        Searches recognized plate records for matching registration numbers.
        
        :param query: Registration search query string (e.g. 'KL07AB1234' or '1234')
        :return: List of matching vehicle records
        """
        if not query:
            return []

        clean_query = query.strip().upper().replace(" ", "")
        matches = []

        for record in self.records.values():
            reg = record.get("registration_number", "").upper()
            if reg != "Unknown" and clean_query in reg:
                matches.append(record)

        return matches
