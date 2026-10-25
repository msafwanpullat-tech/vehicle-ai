import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.services.plate_detector import LicensePlateDetector
from app.services.ocr_service import LicensePlateOCRService
from app.services.plate_registry import PlateRegistryService

def test_ocr_and_plate_detection():
    print("Testing LicensePlateDetector & OCR normalization & Registry search...")

    detector = LicensePlateDetector()
    ocr = LicensePlateOCRService()
    registry = PlateRegistryService()

    # 1. Test OCR Text Normalization
    assert ocr.normalize_registration("kl-07 ab 1234") == "KL07AB1234"
    assert ocr.normalize_registration("  mh 12  cd 9999! ") == "MH12CD9999"
    print("[SUCCESS] OCR text normalization verified.")

    # 2. Test Plate Detector on Synthetic Vehicle Image with Plate Text
    synthetic_vehicle = np.zeros((300, 400, 3), dtype=np.uint8) + 100
    # Draw white license plate box in lower vehicle region
    cv2.rectangle(synthetic_vehicle, (100, 200), (300, 260), (255, 255, 255), -1)
    # Draw dark registration text "KL07AB1234"
    cv2.putText(synthetic_vehicle, "KL07AB1234", (110, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    plate_crop, bbox = detector.detect_plate(synthetic_vehicle)
    assert plate_crop is not None and plate_crop.size > 0, "Plate crop should not be empty!"
    candidates = detector.detect_candidates(synthetic_vehicle)
    assert candidates, "At least one plate candidate should be returned!"
    assert any(candidate_bbox == bbox for _, candidate_bbox in candidates)
    print("[SUCCESS] License plate detection crop algorithm verified.")

    # 2b. Test End-to-End OCR on Detected Plate Crop
    recognized_num, ocr_conf = ocr.recognize_plate(plate_crop)
    assert recognized_num == "KL07AB1234", f"Expected 'KL07AB1234', got '{recognized_num}'"
    assert ocr_conf >= 70.0, f"Expected confidence >= 70.0, got {ocr_conf}"
    print(f"[SUCCESS] End-to-end OCR recognition verified: {recognized_num} ({ocr_conf}%).")

    # 3. Test Plate Registry & Registration Search
    registry.add_or_update_record(
        track_id=17,
        vehicle_type="car",
        registration_number="KL07AB1234",
        confidence=91.0,
        frame_number=81,
        timestamp="00:02:41",
        frame_url="/outputs/frames/frame_81_id17.jpg",
        crop_url="/outputs/plates/plate_id17.jpg"
    )

    registry.add_or_update_record(
        track_id=18,
        vehicle_type="bus",
        registration_number="MH12CD9999",
        confidence=88.5,
        frame_number=120,
        timestamp="00:04:00"
    )

    # Test exact query match
    matches1 = registry.search_registration("KL07AB1234")
    assert len(matches1) == 1, f"Expected 1 match, got {len(matches1)}"
    assert matches1[0]["track_id"] == 17
    assert matches1[0]["vehicle_type"] == "car"
    assert matches1[0]["confidence"] == 91.0

    # Test partial query match
    matches2 = registry.search_registration("1234")
    assert len(matches2) == 1
    assert matches2[0]["registration_number"] == "KL07AB1234"

    # Test no match
    matches3 = registry.search_registration("DL01AA0000")
    assert len(matches3) == 0

    print("[SUCCESS] Registration search engine verified.")
    print("\n[SUCCESS] All Phase 4 OCR & Registration Search unit tests passed successfully!")

if __name__ == "__main__":
    test_ocr_and_plate_detection()
