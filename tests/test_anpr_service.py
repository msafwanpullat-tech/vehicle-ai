import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.services.anpr_service import PlateRecognitionProcessor
from app.services.ocr_service import LicensePlateOCRService


class FakePlateDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect_plate(self, vehicle_roi):
        self.calls += 1
        return vehicle_roi[10:40, 10:90], (10, 10, 90, 40)


class FakeOCRService:
    def __init__(self, text: str, confidence: float, fail: bool = False) -> None:
        self.text = text
        self.confidence = confidence
        self.fail = fail

    def recognize_plate(self, plate_crop):
        if self.fail:
            raise RuntimeError("OCR unavailable")
        return self.text, self.confidence

    @staticmethod
    def normalize_registration(raw_text: str) -> str:
        return LicensePlateOCRService.normalize_registration(raw_text)


class SequencedPlateDetector:
    def __init__(self, crop_sets: list[list[np.ndarray]]) -> None:
        self.crop_sets = crop_sets
        self.calls = 0

    def detect_candidates(self, vehicle_roi):
        crops = self.crop_sets[self.calls]
        self.calls += 1
        return [
            (crop, (0, 0, crop.shape[1], crop.shape[0]))
            for crop in crops
        ]


class CapturingOCRService(FakeOCRService):
    def __init__(self) -> None:
        super().__init__("KL07AB1234", 88.5)
        self.crop_sharpness: list[float] = []

    def recognize_plate(self, plate_crop):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        self.crop_sharpness.append(
            float(cv2.Laplacian(gray, cv2.CV_64F).var())
        )
        return self.text, self.confidence


class PlateRecognitionProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((100, 120, 3), dtype=np.uint8)

    def make_processor(self, text: str, confidence: float, fail: bool = False):
        detector = FakePlateDetector()
        processor = PlateRecognitionProcessor(
            detector=detector,
            ocr_service=FakeOCRService(text, confidence, fail),
            ocr_interval=15,
        )
        return processor, detector

    def test_confident_registration_is_associated_with_track(self) -> None:
        processor, _ = self.make_processor("KL 07 AB 1234", 88.5)

        record, bbox = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 1
        )

        self.assertEqual(record["plate_text"], "KL07AB1234")
        self.assertEqual(record["status"], "Recognized")
        self.assertEqual(record["vehicle_id"], 17)
        self.assertEqual(record["vehicle_class"], "car")
        self.assertEqual(record["ocr_confidence"], 88.5)
        self.assertEqual(bbox, (10, 10, 90, 40))

    def test_crop_quality_rejects_small_and_non_plate_shaped_candidates(self) -> None:
        too_small = np.full((15, 31, 3), 255, dtype=np.uint8)
        wrong_shape = np.full((30, 50, 3), 255, dtype=np.uint8)

        self.assertIsNone(PlateRecognitionProcessor._crop_quality(too_small))
        self.assertIsNone(PlateRecognitionProcessor._crop_quality(wrong_shape))

    def test_selects_sharpest_candidate_and_best_sampled_crop_per_track(self) -> None:
        too_small = np.full((15, 31, 3), 255, dtype=np.uint8)
        blurry = np.full((30, 120, 3), 100, dtype=np.uint8)
        sharp = np.indices((30, 120)).sum(axis=0) % 2
        sharp = np.repeat((sharp * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        detector = SequencedPlateDetector(
            [[too_small, blurry], [sharp], [blurry]]
        )
        ocr = CapturingOCRService()
        processor = PlateRecognitionProcessor(
            detector=detector,
            ocr_service=ocr,
            ocr_interval=15,
        )

        first, _ = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 1
        )
        better, _ = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 16
        )
        inferior, _ = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 31
        )

        self.assertEqual(first["status"], "Recognized")
        self.assertEqual(better["status"], "Recognized")
        self.assertIsNone(inferior)
        self.assertEqual(len(ocr.crop_sharpness), 2)
        self.assertEqual(ocr.crop_sharpness[0], 0.0)
        self.assertGreater(ocr.crop_sharpness[1], ocr.crop_sharpness[0])
        self.assertEqual(detector.calls, 3)

    def test_low_confidence_text_is_not_returned_as_registration(self) -> None:
        processor, _ = self.make_processor("KL 07 AB 1234", 42.0)

        record, _ = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 1
        )

        self.assertEqual(record["plate_text"], "Uncertain")
        self.assertEqual(record["status"], "Uncertain")
        self.assertEqual(record["ocr_confidence"], 42.0)

    def test_ocr_is_throttled_per_track_and_unreadable_is_reported(self) -> None:
        processor, detector = self.make_processor("", 0.0)

        first, _ = processor.process_vehicle(self.frame, (0, 0, 100, 50), 17, "car", 1)
        skipped, _ = processor.process_vehicle(self.frame, (0, 0, 100, 50), 17, "car", 10)
        later, _ = processor.process_vehicle(self.frame, (0, 0, 100, 50), 17, "car", 16)

        self.assertEqual(first["status"], "Unreadable")
        self.assertIsNone(skipped)
        self.assertIsNone(later)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(processor.ocr_attempts, 1)

    def test_ocr_error_is_reported_without_confirmed_plate_text(self) -> None:
        processor, detector = self.make_processor("", 0.0, fail=True)

        record, _ = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 1
        )
        retried, _ = processor.process_vehicle(
            self.frame, (0, 0, 100, 50), 17, "car", 16
        )

        self.assertEqual(record["plate_text"], "OCR unavailable")
        self.assertEqual(record["status"], "OCR unavailable")
        self.assertEqual(retried["status"], "OCR unavailable")
        self.assertEqual(detector.calls, 2)
        self.assertEqual(processor.status, "unavailable")

    def test_debug_mode_saves_candidate_and_ocr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = PlateRecognitionProcessor(
                detector=FakePlateDetector(),
                ocr_service=FakeOCRService("KL 07 AB 1234", 88.5),
                debug_dir=Path(temp_dir),
            )

            record, _ = processor.process_vehicle(
                self.frame, (0, 0, 100, 50), 17, "car", 3
            )

            self.assertEqual(record["status"], "Recognized")
            self.assertTrue(
                (Path(temp_dir) / "frame_000003_track_17_candidate.jpg").is_file()
            )
            metadata = json.loads(
                (Path(temp_dir) / "frame_000003_track_17.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["vehicle_id"], 17)
            self.assertEqual(metadata["selected_result"]["plate_text"], "KL07AB1234")

    def test_analysis_saves_vehicle_and_plate_images_with_video_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            processor = PlateRecognitionProcessor(
                detector=FakePlateDetector(),
                ocr_service=FakeOCRService("KL 07 AB 1234", 88.5),
                analysis_dir=base / "analysis_test",
                session_dir=base / "session_test",
            )
            processor.set_video_fps(10)

            processor.process_vehicle(self.frame, (0, 0, 100, 50), 17, "car", 61)
            processor.save_analysis_manifest("/output.mp4", "/source.mp4")

            self.assertTrue(
                (base / "analysis_test" / "frame_000061_track_17_vehicle.jpg").is_file()
            )
            self.assertTrue(
                (base / "analysis_test" / "frame_000061_track_17_plate.jpg").is_file()
            )
            manifest = json.loads(
                (base / "session_test" / "analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["detections"][0]["timestamp"], "00:00:06")
            self.assertEqual(manifest["detections"][0]["timestamp_seconds"], 6.0)

    def test_tracking_event_records_crossing_and_whether_it_was_counted(self) -> None:
        processor, _ = self.make_processor("KL 07 AB 1234", 88.5)
        processor.set_video_fps(10)

        processor.add_tracking_event(61, 17, "car", True, "down", False)

        event = processor.tracking_events[0]
        self.assertEqual(event["timestamp"], "00:00:06")
        self.assertTrue(event["line_crossing"])
        self.assertEqual(event["crossing_direction"], "down")
        self.assertFalse(event["counted_for_total"])


if __name__ == "__main__":
    unittest.main()
