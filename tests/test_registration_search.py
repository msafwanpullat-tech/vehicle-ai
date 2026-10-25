import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.routes.counting_routes import (
    RegistrationSearchRequest,
    search_vehicle_registration,
)
from app.services.registration_search import (
    format_video_timestamp,
    is_valid_registration,
    normalize_registration,
    search_detections,
)


def detection(
    vehicle_id: int,
    text: str,
    status: str = "Recognized",
    confidence: float = 90,
    timestamp_seconds: float = 1.0,
) -> dict[str, object]:
    return {
        "vehicle_id": vehicle_id,
        "vehicle_class": "car",
        "ocr_text": text,
        "status": status,
        "ocr_confidence": confidence,
        "timestamp_seconds": timestamp_seconds,
        "timestamp": "00:00:01",
        "vehicle_image_url": f"/vehicle-{vehicle_id}.jpg",
        "plate_image_url": f"/plate-{vehicle_id}.jpg",
    }


class RegistrationSearchTests(unittest.TestCase):
    def test_normalizes_case_spaces_and_hyphens_and_validates_indian_format(self) -> None:
        normalized = normalize_registration(" kl-07 ab 1234 ")

        self.assertEqual(normalized, "KL07AB1234")
        self.assertTrue(is_valid_registration(normalized))
        self.assertFalse(is_valid_registration("KL@07AB1234"))
        self.assertFalse(is_valid_registration("not a registration"))

    def test_returns_multiple_exact_tracking_occurrences(self) -> None:
        result = search_detections(
            "KL 07 AB 1234",
            [
                detection(7, "KL07AB1234", timestamp_seconds=2),
                detection(11, "KL07AB1234", timestamp_seconds=18),
                detection(12, "KL07AB1235", status="Uncertain"),
            ],
        )

        self.assertEqual(
            [item["vehicle_id"] for item in result["exact_matches"]], [7, 11]
        )
        self.assertEqual(result["exact_matches"][0]["match_type"], "Exact")
        self.assertFalse(result["not_found"])

    def test_similar_ocr_is_uncertain_and_not_an_exact_match(self) -> None:
        item = detection(
            9,
            "KL07AB1235",
            status="Uncertain",
            confidence=84,
        )
        item["ocr_candidates"] = [
            {"text": "KL07AB1235", "confidence": 84}
        ]

        result = search_detections("KL07AB1234", [item])

        self.assertEqual(result["exact_matches"], [])
        self.assertEqual(len(result["uncertain_candidates"]), 1)
        self.assertEqual(
            result["uncertain_candidates"][0]["match_type"],
            "Approximate OCR candidate",
        )
        self.assertFalse(result["not_found"])

    def test_no_match_is_explicit(self) -> None:
        result = search_detections("KL07AB1234", [detection(1, "MH12CD9999")])

        self.assertEqual(result["exact_matches"], [])
        self.assertEqual(result["uncertain_candidates"], [])
        self.assertTrue(result["not_found"])

    def test_timestamp_uses_frame_number_and_fps(self) -> None:
        seconds, timestamp = format_video_timestamp(36001, 10)

        self.assertEqual(seconds, 3600)
        self.assertEqual(timestamp, "01:00:00")

    def test_search_endpoint_uses_analysis_manifest_and_returns_safe_urls(self) -> None:
        job_id = "a" * 32
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_dir = Path(temp_dir) / ".analysis_sessions" / job_id
            manifest_dir.mkdir(parents=True)
            output_dir = Path(temp_dir) / "outputs" / "counting"
            analysis_dir = output_dir / f"analysis_{job_id}"
            analysis_dir.mkdir(parents=True)
            output_path = analysis_dir / f"vehicle_count_{job_id}.mp4"
            source_path = analysis_dir / "source.mp4"
            output_path.touch()
            source_path.touch()
            manifest = {
                "analysis_id": job_id,
                "output_video_url": (
                    f"/outputs/counting/analysis_{job_id}/{output_path.name}"
                ),
                "source_video_url": (
                    f"/outputs/counting/analysis_{job_id}/{source_path.name}"
                ),
                "detections": [
                    detection(17, "KL07AB1234", timestamp_seconds=4)
                ],
            }
            (manifest_dir / "analysis.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with (
                patch("app.routes.counting_routes.PROJECT_ROOT", Path(temp_dir)),
                patch("app.routes.counting_routes.OUTPUT_DIR", output_dir),
                patch(
                    "app.routes.counting_routes.get_analysis_manifest",
                    return_value=manifest,
                ),
                patch("app.routes.counting_routes.save_manifest"),
            ):
                result = search_vehicle_registration(
                    job_id,
                    RegistrationSearchRequest(
                        registration_number=" kl-07 ab 1234 "
                    ),
                )
        self.assertEqual(result["query"], "KL07AB1234")
        self.assertEqual(len(result["exact_matches"]), 1)
        self.assertEqual(len(result["exact_matches"]), 1)
        self.assertEqual(result["output_video_url"], manifest["output_video_url"])
        self.assertEqual(result["source_video_url"], manifest["source_video_url"])
        self.assertNotIn(temp_dir, json.dumps(result))

    def test_search_endpoint_rejects_invalid_registration(self) -> None:
        with self.assertRaises(HTTPException) as error:
            search_vehicle_registration(
                "a" * 32,
                RegistrationSearchRequest(registration_number="KL@07AB1234"),
            )

        self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
