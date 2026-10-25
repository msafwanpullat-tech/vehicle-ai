"""Track and count vehicles crossing a configurable horizontal line."""

import argparse
import logging
import math
import os
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from app.services.anpr_service import PlateRecognitionProcessor


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT_DIR / "models" / "indian_traffic_ft_best.pt"
DEFAULT_VIDEO = ROOT_DIR / "videos" / "sample_traffic_moving.mp4"
DEFAULT_OUTPUT = ROOT_DIR / "runs" / "vehicle_counting_output.mp4"

logger = logging.getLogger("vehicle_counting")


class InvalidVideoError(RuntimeError):
    """Raised when the supplied file cannot be decoded as a video."""


def validate_decoded_frame_count(expected_frames: int, frames_processed: int) -> None:
    if frames_processed < 2:
        raise InvalidVideoError(
            "The file did not contain enough decodable video frames. It may be corrupt "
            "or use an unsupported codec; try re-encoding it as H.264 MP4 or MJPEG AVI."
        )
    if expected_frames > 0 and frames_processed != expected_frames:
        logger.warning(
            "Video metadata reports %d frames, but the decoder returned %d; "
            "using the frames successfully decoded.",
            expected_frames,
            frames_processed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count uniquely tracked vehicles crossing a horizontal video line."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO model weights")
    parser.add_argument("--input", type=Path, default=DEFAULT_VIDEO, help="Input traffic video")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MP4 path")
    line_options = parser.add_mutually_exclusive_group()
    line_options.add_argument(
        "--line-position",
        type=float,
        default=55.6,
        help="Counting-line height as a percentage of frame height (0-100; default: 55.6)",
    )
    line_options.add_argument(
        "--line-ratio",
        type=float,
        help="Legacy fraction-based line position (0.0-1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each detection's ID, class, center, and line-crossing status",
    )
    args = parser.parse_args()
    if args.line_ratio is None:
        if not math.isfinite(args.line_position) or not 0.0 <= args.line_position <= 100.0:
            parser.error("--line-position must be a finite percentage between 0 and 100")
    elif not math.isfinite(args.line_ratio) or not 0.0 <= args.line_ratio <= 1.0:
        parser.error("--line-ratio must be a finite number between 0.0 and 1.0")
    return args


def crossing_direction(
    previous_center: tuple[int, int],
    center: tuple[int, int],
    line_y: int,
) -> str | None:
    previous_y = previous_center[1]
    current_y = center[1]
    if previous_y < line_y <= current_y:
        return "down"
    if previous_y > line_y >= current_y:
        return "up"
    return None


class VehicleCrossingCounter:
    def __init__(self, class_counts: dict[str, int]) -> None:
        self.class_counts = class_counts
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.track_class_votes: dict[int, Counter[str]] = {}
        self.counted_ids: set[int] = set()

    @property
    def total_count(self) -> int:
        return sum(self.class_counts.values())

    def observe(
        self,
        track_id: int | None,
        detected_class_name: str,
        center: tuple[int, int],
        line_y: int,
    ) -> tuple[str, str | None, bool]:
        class_name = detected_class_name
        crossed_direction = None
        counted_now = False

        if track_id is not None:
            class_votes = self.track_class_votes.setdefault(track_id, Counter())
            class_votes[detected_class_name] += 1
            class_name = class_votes.most_common(1)[0][0]

            previous_center = self.previous_centers.get(track_id)
            if previous_center is not None:
                crossed_direction = crossing_direction(previous_center, center, line_y)
                if crossed_direction is not None and track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.class_counts.setdefault(class_name, 0)
                    self.class_counts[class_name] += 1
                    counted_now = True

            self.previous_centers[track_id] = center

        return class_name, crossed_direction, counted_now


def draw_count_overlay(
    frame: cv2.typing.MatLike,
    total_count: int,
    class_counts: dict[str, int],
) -> None:
    height, width = frame.shape[:2]
    columns = 2 if width >= 640 else 1
    rows = (len(class_counts) + columns - 1) // columns
    row_height = max(14, min(20, height // max(rows + 2, 1)))
    font_scale = 0.42 if width >= 640 else 0.38
    panel_height = min(height, 30 + rows * row_height)
    panel_width = min(width, 560 if columns == 2 else 400)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        f"Total vehicles: {total_count}",
        (8, min(22, height - 1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    column_width = panel_width // columns
    for index, (class_name, count) in enumerate(class_counts.items()):
        column = index // rows if rows else 0
        row = index % rows if rows else 0
        x = 8 + column * column_width
        y = 30 + row * row_height
        if y >= height:
            continue
        cv2.putText(
            frame,
            f"{class_name}: {count}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def process_video(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    line_position_percent: float,
    debug: bool = False,
    browser_compatible: bool = False,
    anpr_processor: PlateRecognitionProcessor | None = None,
) -> tuple[int, dict[str, int]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    model = YOLO(str(model_path))
    if isinstance(model.names, dict):
        class_names = {int(class_id): str(name) for class_id, name in model.names.items()}
    else:
        class_names = {index: str(name) for index, name in enumerate(model.names)}
    class_counts = {name: 0 for name in class_names.values()}
    crossing_counter = VehicleCrossingCounter(class_counts)

    capture = cv2.VideoCapture(str(input_path))
    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    try:
        if not capture.isOpened():
            raise InvalidVideoError(
                "The video could not be opened by the installed decoders. It may use an "
                "unsupported codec; try re-encoding it as H.264 MP4 or MJPEG AVI."
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise InvalidVideoError(f"Input video has invalid frame dimensions: {input_path}")
        if not math.isfinite(fps) or fps <= 0:
            logger.warning("Input video has invalid FPS metadata; using 30 FPS for output.")
            fps = 30.0
        if anpr_processor is not None:
            anpr_processor.set_video_fps(fps)

        line_y = min(height - 1, int(height * line_position_percent / 100))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if browser_compatible and os.name == "nt":
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.CAP_MSMF,
                cv2.VideoWriter_fourcc(*"H264"),
                fps,
                (width, height),
            )
        else:
            output_codec = "VP80" if output_path.suffix.lower() == ".webm" else "mp4v"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*output_codec),
                fps,
                (width, height),
            )
        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV could not create the output video with the required browser-compatible codec."
                if browser_compatible
                else f"OpenCV could not create output video: {output_path}"
            )

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None:
                raise InvalidVideoError(
                    "The video decoder returned an invalid frame. The file may be damaged "
                    "or use an unsupported codec; try re-encoding it as H.264 MP4 or MJPEG AVI."
                )
            frames_processed += 1

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                device="cpu",
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                track_ids = boxes.id
                for index, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0].item())
                    detected_class_name = class_names.get(class_id, f"class_{class_id}")
                    color = (40, 220, 40)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    track_id = (
                        int(track_ids[index].item())
                        if track_ids is not None and index < len(track_ids)
                        else None
                    )

                    if anpr_processor is not None and track_id is not None:
                        plate_record, plate_bbox = anpr_processor.process_vehicle(
                            frame,
                            (x1, y1, x2, y2),
                            track_id,
                            detected_class_name,
                            frames_processed,
                        )
                        if plate_record is not None and plate_bbox is not None:
                            px1, py1, px2, py2 = plate_bbox
                            cv2.rectangle(
                                frame,
                                (x1 + px1, y1 + py1),
                                (x1 + px2, y1 + py2),
                                (255, 180, 40),
                                2,
                            )
                            plate_label = (
                                f"{plate_record['plate_text']} "
                                f"{plate_record['ocr_confidence']:.0f}%"
                            )
                            cv2.putText(
                                frame,
                                plate_label,
                                (x1 + px1, max(18, y1 + py1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 180, 40),
                                1,
                                cv2.LINE_AA,
                            )

                    class_name, crossed_direction, counted_now = crossing_counter.observe(
                        track_id, detected_class_name, center, line_y
                    )
                    if anpr_processor is not None and track_id is not None:
                        anpr_processor.add_tracking_event(
                            frames_processed,
                            track_id,
                            class_name,
                            crossed_direction is not None,
                            crossed_direction,
                            counted_now,
                        )
                    if track_id is not None:
                        label = f"{class_name} | ID: {track_id}"
                    else:
                        label = class_name

                    if debug:
                        if track_id is None:
                            crossing_status = "unknown (no tracking ID; not counted)"
                        elif crossed_direction is None:
                            crossing_status = "no"
                        elif counted_now:
                            crossing_status = f"yes ({crossed_direction}; counted)"
                        else:
                            crossing_status = f"yes ({crossed_direction}; already counted)"
                        logger.info(
                            "frame=%d ID=%s class=%s center=(%d,%d) "
                            "line_y=%d crossed=%s",
                            frames_processed,
                            track_id if track_id is not None else "unavailable",
                            class_name,
                            center[0],
                            center[1],
                            line_y,
                            crossing_status,
                        )
                        if track_id is not None and class_name != detected_class_name:
                            logger.info(
                                "frame=%d ID=%d observed_class=%s "
                                "track_class_majority=%s",
                                frames_processed,
                                track_id,
                                detected_class_name,
                                class_name,
                            )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            draw_count_overlay(frame, crossing_counter.total_count, class_counts)
            cv2.line(frame, (0, line_y), (width - 1, line_y), (0, 0, 255), 2)
            writer.write(frame)

        validate_decoded_frame_count(expected_frames, frames_processed)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output video was not written successfully: {output_path}")

    verification_capture = cv2.VideoCapture(str(output_path))
    try:
        if not verification_capture.isOpened():
            raise RuntimeError(f"Output video cannot be reopened: {output_path}")
        success, _ = verification_capture.read()
        if not success:
            raise RuntimeError(f"Output video contains no readable frames: {output_path}")
    finally:
        verification_capture.release()

    return crossing_counter.total_count, class_counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        total_count, class_counts = process_video(
            model_path=args.model,
            input_path=args.input,
            output_path=args.output,
            line_position_percent=(
                args.line_ratio * 100 if args.line_ratio is not None else args.line_position
            ),
            debug=args.debug,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, cv2.error) as error:
        logger.error("%s", error)
        return 1

    print(f"Total vehicles counted: {total_count}")
    print("Class-wise counts:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    print(f"Output video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
