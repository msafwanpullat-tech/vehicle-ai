import re
from collections.abc import Iterable
from typing import Any

INDIAN_REGISTRATION_PATTERN = re.compile(
    r"^(?:[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}|BH\d{2}[A-Z]{2}\d{4})$"
)


def normalize_registration(value: str) -> str:
    return re.sub(r"[\s-]+", "", value.upper())


def is_valid_registration(value: str) -> bool:
    return bool(INDIAN_REGISTRATION_PATTERN.fullmatch(normalize_registration(value)))


def format_video_timestamp(frame_number: int, fps: float) -> tuple[float, str]:
    if fps <= 0:
        raise ValueError("FPS must be positive to calculate a video timestamp.")
    seconds = max(0.0, (frame_number - 1) / fps)
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return seconds, f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _edit_distance_at_most_one(left: str, right: str) -> int | None:
    if abs(len(left) - len(right)) > 1:
        return None
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left

    left_index = right_index = differences = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            left_index += 1
            right_index += 1
            continue
        differences += 1
        if differences > 1:
            return None
        if len(left) == len(right):
            left_index += 1
        right_index += 1

    if left_index < len(left) or right_index < len(right):
        differences += 1
    return differences if differences <= 1 else None


def search_detections(
    query: str, detections: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    normalized_query = normalize_registration(query)
    exact_by_track: dict[int, dict[str, Any]] = {}
    uncertain_by_track: dict[int, tuple[int, dict[str, Any]]] = {}

    for detection in detections:
        track_id = int(detection["vehicle_id"])
        status = str(detection.get("status", ""))
        confidence = float(detection.get("ocr_confidence", 0))
        candidates = detection.get("ocr_candidates") or [
            {
                "text": detection.get("ocr_text", ""),
                "confidence": confidence,
            }
        ]
        selected_text = normalize_registration(str(detection.get("ocr_text", "")))
        is_confirmed_exact = status == "Recognized" and selected_text == normalized_query

        if is_confirmed_exact:
            previous = exact_by_track.get(track_id)
            if previous is None or confidence > float(previous["ocr_confidence"]):
                result = dict(detection)
                result["ocr_text"] = selected_text
                result["match_type"] = "Exact"
                exact_by_track[track_id] = result
            continue

        nearest_distance: int | None = None
        candidate_confidence = confidence
        for candidate in candidates:
            candidate_text = normalize_registration(str(candidate.get("text", "")))
            candidate_conf = float(candidate.get("confidence", confidence))
            if not candidate_text:
                continue
            distance = _edit_distance_at_most_one(normalized_query, candidate_text)
            if distance is None or candidate_conf < 50:
                continue
            if nearest_distance is None or (
                distance,
                -candidate_conf,
            ) < (nearest_distance, -candidate_confidence):
                nearest_distance = distance
                candidate_confidence = candidate_conf
                selected_text = candidate_text
        if nearest_distance is not None:
            previous = uncertain_by_track.get(track_id)
            if previous is None or (
                nearest_distance,
                -candidate_confidence,
            ) < (previous[0], -float(previous[1]["ocr_confidence"])):
                result = dict(detection)
                result["ocr_confidence"] = candidate_confidence
                result["ocr_text"] = selected_text
                result["match_type"] = "Approximate OCR candidate"
                uncertain_by_track[track_id] = (nearest_distance, result)

    exact_matches = sorted(
        exact_by_track.values(),
        key=lambda detection: (float(detection["timestamp_seconds"]), int(detection["vehicle_id"])),
    )
    uncertain_candidates = [
        detection
        for _, detection in sorted(
            uncertain_by_track.values(),
            key=lambda item: (
                item[0],
                float(item[1]["timestamp_seconds"]),
                int(item[1]["vehicle_id"]),
            ),
        )
    ]
    return {
        "query": normalized_query,
        "exact_matches": exact_matches,
        "uncertain_candidates": uncertain_candidates,
        "not_found": not exact_matches and not uncertain_candidates,
    }
