import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger("vehicle_ai.plate_detector")

class LicensePlateDetector:
    """
    Computer Vision Service for detecting and cropping license plate regions from vehicle ROIs.
    Combines morphological gradient analysis, aspect-ratio filtering, and contour geometry.
    """

    def __init__(self):
        # Rectangular kernel for morphological blackhat filtering
        self.rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self.square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect_candidates(
        self, vehicle_roi: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """Detect plausible plate crops from a BGR vehicle ROI, ranked by geometry."""
        if vehicle_roi is None or vehicle_roi.size == 0:
            return []

        vh, vw, _ = vehicle_roi.shape
        if vh < 20 or vw < 20:
            return []

        # Focus on the lower half to avoid badges and grille emblems.
        roi_top = int(vh * 0.45)
        roi_bottom = int(vh * 0.98)
        roi_left = int(vw * 0.05)
        roi_right = int(vw * 0.95)

        search_area = vehicle_roi[roi_top:roi_bottom, roi_left:roi_right]
        sh, sw, _ = search_area.shape
        if sh < 15 or sw < 30:
            search_area = vehicle_roi
            roi_top, roi_left = 0, 0
            sh, sw, _ = vehicle_roi.shape

        gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)

        # Apply Blackhat morphological operation to reveal license plate characters
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rect_kernel)

        # Compute Sobel X gradient to detect vertical edges of license plate numbers
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        if max_val > min_val:
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        else:
            grad_x = grad_x.astype("uint8")

        # Blur and threshold to group character regions
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to connect adjacent characters into a license plate rectangle
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.rect_kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[
            tuple[float, np.ndarray, tuple[int, int, int, int]]
        ] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h)
            area = w * h

            if 1.2 <= aspect_ratio <= 10.0 and area > 150:
                aspect_score = max(0.1, 1.0 - abs(aspect_ratio - 3.0) / 5.0)
                score = area * aspect_score

                pad_x = max(4, int(w * 0.08))
                pad_y = max(4, int(h * 0.15))
                px1 = max(0, roi_left + x - pad_x)
                py1 = max(0, roi_top + y - pad_y)
                px2 = min(vw, roi_left + x + w + pad_x)
                py2 = min(vh, roi_top + y + h + pad_y)
                bbox = (px1, py1, px2, py2)
                crop = vehicle_roi[py1:py2, px1:px2]
                if crop.size:
                    candidates.append((score, crop, bbox))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [(crop, bbox) for _, crop, bbox in candidates[:5]]

    def detect_plate(
        self, vehicle_roi: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Return the highest geometry-ranked plate crop for compatibility."""
        candidates = self.detect_candidates(vehicle_roi)
        if not candidates:
            return None, None

        return candidates[0]
