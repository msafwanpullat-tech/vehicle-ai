import re
import cv2
import numpy as np
import logging
from typing import Tuple, Optional

from app.services.registration_search import (
    INDIAN_REGISTRATION_PATTERN,
    is_valid_registration,
)

logger = logging.getLogger("vehicle_ai.ocr_service")

# Lazy import for EasyOCR
_easyocr_reader = None


def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        logger.info("Initializing EasyOCR CPU reader...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "7",
    "S": "5",
    "B": "8",
    "G": "6",
}
TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
}
NOISE_WORDS = {
    "CARRIER",
    "GOODS",
    "PERMIT",
    "ALL",
    "INDIA",
    "STOP",
    "HONDA",
    "SUZUKI",
    "TOYOTA",
    "HYUNDAI",
    "TATA",
    "MAHINDRA",
    "POLICE",
    "GOVT",
    "GOVERNMENT",
    "SPEED",
}


def repair_plate(text: str) -> str:
    """Attempts to disambiguate OCR character confusion (e.g. O vs 0, Z vs 7/2, I vs 1)

    to match the standard Indian registration pattern.
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if INDIAN_REGISTRATION_PATTERN.fullmatch(cleaned):
        return cleaned

    n = len(cleaned)
    if not (7 <= n <= 11):
        return cleaned

    for dist_len in (2, 1):
        for series_len in (2, 1, 3, 0):
            num_len = n - 2 - dist_len - series_len
            if 1 <= num_len <= 4:
                state_raw = cleaned[:2]
                dist_raw = cleaned[2 : 2 + dist_len]
                series_raw = cleaned[2 + dist_len : 2 + dist_len + series_len]
                num_raw = cleaned[2 + dist_len + series_len :]

                dist_options = [""]
                for ch in dist_raw:
                    cand_digits = ["7", "2"] if ch == "Z" else [TO_DIGIT.get(ch, ch)]
                    dist_options = [
                        prev + d for prev in dist_options for d in cand_digits
                    ]

                state_fixed = "".join(TO_LETTER.get(c, c) for c in state_raw)
                series_fixed = "".join(TO_LETTER.get(c, c) for c in series_raw)

                num_options = [""]
                for ch in num_raw:
                    cand_digits = ["7", "2"] if ch == "Z" else [TO_DIGIT.get(ch, ch)]
                    num_options = [
                        prev + d for prev in num_options for d in cand_digits
                    ]

                for d_cand in dist_options:
                    for n_cand in num_options:
                        candidate = f"{state_fixed}{d_cand}{series_fixed}{n_cand}"
                        if INDIAN_REGISTRATION_PATTERN.fullmatch(candidate):
                            return candidate

    return cleaned


class LicensePlateOCRService:
    """OCR Engine for License Plate Recognition.

    Includes image preprocessing (resizing, grayscale conversion, and CLAHE)
    and text normalization (uppercase, alphanumeric filtering, confidence estimation).
    """

    def __init__(self) -> None:
        self.last_debug_results: list[dict[str, str | float]] = []

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Preprocesses license plate crop image for optimal OCR character recognition:

        1. Resizes a small crop to a height suitable for OCR.
        2. Grayscale conversion.
        3. CLAHE contrast enhancement.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        target_h = max(160, min(320, h * 4))
        aspect = float(w) / float(h)
        target_w = max(80, int(target_h * aspect))

        resized = cv2.resize(
            plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def prepare_plate_variants(
        self, plate_crop: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        enhanced = self.preprocess_plate(plate_crop)
        if enhanced is None or enhanced.size == 0:
            return []
        return [
            ("clahe_gray", enhanced),
            ("color_upscaled", self._resize_color(plate_crop)),
        ]

    @staticmethod
    def _resize_color(plate_crop: np.ndarray) -> np.ndarray:
        height, width = plate_crop.shape[:2]
        target_height = max(160, min(320, height * 4))
        target_width = max(80, int(width * target_height / height))
        return cv2.resize(
            plate_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )

    def recognize_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Recognizes registration text from license plate image.

        :param plate_crop: BGR license plate crop image
        :return: (normalized_registration_number, confidence_percentage)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "Unknown", 0.0

        reader = get_easyocr_reader()
        self.last_debug_results = []
        if reader is None:
            return "", 0.0

        for variant_name, variant in self.prepare_plate_variants(plate_crop):
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                contrast_ths=0.1,
                adjust_contrast=0.5,
                decoder="greedy",
            )
            variant_boxes = []
            for result in results:
                raw_text = str(result[1])
                confidence = max(0.0, min(1.0, float(result[2]))) * 100.0
                poly = result[0]
                y_center = sum(p[1] for p in poly) / 4.0
                x_center = sum(p[0] for p in poly) / 4.0
                h_box = max(p[1] for p in poly) - min(p[1] for p in poly)
                variant_boxes.append(
                    {
                        "raw_text": raw_text,
                        "confidence": confidence,
                        "y": y_center,
                        "x": x_center,
                        "h": h_box,
                    }
                )
                self.last_debug_results.append(
                    {
                        "variant": variant_name,
                        "raw_text": raw_text,
                        "normalized_text": self.normalize_registration(raw_text),
                        "confidence": round(confidence, 1),
                    }
                )

            # Combine multi-box / multi-line plate detections in reading order
            if len(variant_boxes) > 1:
                avg_h = sum(b["h"] for b in variant_boxes) / len(variant_boxes)
                sorted_boxes = sorted(
                    variant_boxes,
                    key=lambda b: (
                        round(b["y"] / max(1.0, avg_h * 0.6)),
                        b["x"],
                    ),
                )
                combined_raw = "".join(b["raw_text"] for b in sorted_boxes)
                combined_conf = sum(b["confidence"] for b in sorted_boxes) / len(
                    variant_boxes
                )
                self.last_debug_results.append(
                    {
                        "variant": f"{variant_name}_combined",
                        "raw_text": combined_raw,
                        "normalized_text": self.normalize_registration(combined_raw),
                        "confidence": round(combined_conf, 1),
                    }
                )

        if not self.last_debug_results:
            return "Unknown", 0.0

        def candidate_score(cand: dict[str, str | float]) -> float:
            raw = str(cand.get("raw_text", ""))
            conf = float(cand.get("confidence", 0.0))
            norm = self.normalize_registration(raw)
            repaired = repair_plate(norm)

            if any(w in norm for w in NOISE_WORDS):
                return conf - 500.0

            if is_valid_registration(repaired):
                return conf + 1000.0

            if 7 <= len(norm) <= 11:
                return conf + 50.0

            return conf

        best_result = max(self.last_debug_results, key=candidate_score)
        raw_text = str(best_result["raw_text"])
        confidence = float(best_result["confidence"])

        normalized_text = self.normalize_registration(raw_text)
        repaired_text = repair_plate(normalized_text)

        if is_valid_registration(repaired_text):
            final_text = repaired_text
        else:
            final_text = normalized_text

        if len(final_text) < 4:
            return "Unknown", 0.0

        return final_text, round(confidence, 1)

    @staticmethod
    def normalize_registration(raw_text: str) -> str:
        """Normalizes OCR output text:

        1. Uppercase conversion.
        2. Removes non-alphanumeric characters.
        3. Removes spaces & symbols.
        """
        if not raw_text:
            return ""

        # Convert to uppercase and strip spaces/symbols
        clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        return clean
