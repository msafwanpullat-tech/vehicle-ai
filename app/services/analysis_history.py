import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import PROJECT_ROOT, SessionLocal
from app.models.analysis_history import AnalysisRecord

logger = logging.getLogger("vehicle_ai.analysis_history")
SESSIONS_DIR = PROJECT_ROOT / ".analysis_sessions"
COUNTING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "counting"
ANALYSIS_ID_LENGTH = 32


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True))


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _manifest_datetime(manifest: dict[str, Any], manifest_path: Path) -> datetime:
    value = manifest.get("created_at")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        parsed = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _path_from_public_url(value: Any, analysis_id: str) -> Path | None:
    if not isinstance(value, str) or not value.startswith("/outputs/counting/"):
        return None
    relative = value.split("?", 1)[0].removeprefix("/outputs/counting/")
    candidate = (COUNTING_OUTPUT_DIR / relative).resolve()
    try:
        candidate.relative_to(COUNTING_OUTPUT_DIR.resolve())
    except ValueError:
        return None
    is_analysis_file = f"analysis_{analysis_id}" in candidate.parts
    is_legacy_output = candidate.parent == COUNTING_OUTPUT_DIR.resolve() and candidate.name in {
        f"vehicle_count_{analysis_id}.mp4",
        f"vehicle_count_{analysis_id}.webm",
    }
    if not (is_analysis_file or is_legacy_output):
        return None
    return candidate


def _manifest_artifacts(manifest: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for field, key in (
        ("source_video_url", "source_video"),
        ("output_video_url", "annotated_video"),
    ):
        value = manifest.get(field)
        path = _path_from_public_url(value, analysis_id)
        if value is not None and path is None:
            logger.warning(
                "Analysis %s contains an invalid %s path reference.", analysis_id, key
            )
        if path is not None:
            artifacts[key] = _relative_path(path)
            if not path.is_file():
                logger.warning(
                    "Analysis %s references a missing %s artifact.", analysis_id, key
                )

    vehicle_images: list[str] = []
    plate_images: list[str] = []
    detections = manifest.get("detections", [])
    if isinstance(detections, list):
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            for field, collection in (
                ("vehicle_image_url", vehicle_images),
                ("plate_image_url", plate_images),
            ):
                value = detection.get(field)
                image_path = _path_from_public_url(value, analysis_id)
                if value is not None and image_path is None:
                    logger.warning(
                        "Analysis %s contains an invalid image path reference.",
                        analysis_id,
                    )
                if image_path is not None:
                    collection.append(_relative_path(image_path))
                    if not image_path.is_file():
                        logger.warning(
                            "Analysis %s references a missing image artifact.", analysis_id
                        )
    artifacts["vehicle_screenshots"] = vehicle_images
    artifacts["plate_crops"] = plate_images
    session_manifest = SESSIONS_DIR / analysis_id / "analysis.json"
    if session_manifest.is_file():
        artifacts["manifest"] = _relative_path(session_manifest)
    return artifacts


def create_analysis(
    analysis_id: str,
    filename: str,
    line_position_percent: float,
    analysis_dir: Path,
    session_dir: Path,
    output_file: str,
) -> None:
    with SessionLocal() as session:
        session.add(
            AnalysisRecord(
                id=analysis_id,
                filename=Path(filename).name[:255],
                status="processing",
                line_position_percent=line_position_percent,
                artifact_paths={
                    "analysis_dir": _relative_path(analysis_dir),
                    "session_dir": _relative_path(session_dir),
                    "annotated_video": _relative_path(analysis_dir / output_file),
                },
                export_metadata={},
            )
        )
        session.commit()


def complete_analysis(
    analysis_id: str,
    manifest: dict[str, Any],
    duration_seconds: float | None,
) -> None:
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise LookupError("Analysis history record is missing.")
        record.status = "completed"
        record.completed_at = datetime.now(timezone.utc)
        record.duration_seconds = duration_seconds
        record.line_position_percent = manifest.get("line_position_percent")
        record.total_count = manifest.get("total_line_crossings")
        record.class_counts = _json_value(manifest.get("class_counts") or {})
        record.tracking_events = _json_value(manifest.get("tracking_events") or [])
        record.anpr_records = _json_value(manifest.get("detections") or [])
        record.search_results = _json_value(manifest.get("search_results") or [])
        record.manifest = _json_value(manifest)
        record.artifact_paths = {
            **(record.artifact_paths or {}),
            **_manifest_artifacts(manifest, analysis_id),
        }
        record.failure_message = None
        record.export_metadata = {
            "available": [
                "csv",
                "excel",
                "pdf",
                "zip",
                "vehicle_screenshots",
                "plate_crops",
            ]
        }
        session.commit()


def fail_analysis(analysis_id: str, message: str) -> None:
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            return
        record.status = "failed"
        record.completed_at = datetime.now(timezone.utc)
        record.failure_message = message[:512]
        session.commit()


def save_manifest(analysis_id: str, manifest: dict[str, Any]) -> None:
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise LookupError("Analysis history record is missing.")
        record.manifest = _json_value(manifest)
        record.tracking_events = _json_value(manifest.get("tracking_events") or [])
        record.anpr_records = _json_value(manifest.get("detections") or [])
        record.search_results = _json_value(manifest.get("search_results") or [])
        record.artifact_paths = {
            **(record.artifact_paths or {}),
            **_manifest_artifacts(manifest, analysis_id),
        }
        session.commit()


def get_analysis(analysis_id: str) -> AnalysisRecord | None:
    with SessionLocal() as session:
        return session.get(AnalysisRecord, analysis_id)


def get_analysis_manifest(analysis_id: str) -> dict[str, Any] | None:
    record = get_analysis(analysis_id)
    if record is None or not isinstance(record.manifest, dict):
        return None
    return record.manifest


def list_analyses(
    search: str,
    status: str,
    date_from: date | None,
    date_to: date | None,
    sort: str,
    page: int,
    page_size: int,
    db: Session | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if db is not None:
        return _list_analyses(db, search, status, date_from, date_to, sort, page, page_size)
    with SessionLocal() as session:
        return _list_analyses(session, search, status, date_from, date_to, sort, page, page_size)


def _list_analyses(
    session: Session,
    search: str,
    status: str,
    date_from: date | None,
    date_to: date | None,
    sort: str,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    statement = select(AnalysisRecord)
    if search:
        statement = statement.where(AnalysisRecord.filename.ilike(f"%{search}%"))
    if status in {"processing", "completed", "failed"}:
        statement = statement.where(AnalysisRecord.status == status)
    if date_from:
        statement = statement.where(
            AnalysisRecord.created_at
            >= datetime.combine(date_from, time.min, timezone.utc)
        )
    if date_to:
        statement = statement.where(
            AnalysisRecord.created_at
            < datetime.combine(date_to + timedelta(days=1), time.min, timezone.utc)
        )
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    ordering = (
        AnalysisRecord.created_at.asc()
        if sort == "oldest"
        else AnalysisRecord.created_at.desc()
    )
    records = session.scalars(
        statement.order_by(ordering, AnalysisRecord.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return [
        {
            "analysis_id": record.id,
            "filename": record.filename,
            "created_at": record.created_at.isoformat(),
            "status": record.status,
            "total_count": record.total_count,
            "failure_message": record.failure_message,
            "duration_seconds": record.duration_seconds,
        }
        for record in records
    ], total


def delete_analysis_record(analysis_id: str) -> bool:
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            return False
        session.delete(record)
        session.commit()
        return True


def import_legacy_manifests(dry_run: bool = True) -> dict[str, int]:
    imported = 0
    skipped = 0
    invalid = 0
    with SessionLocal() as session:
        for session_directory in SESSIONS_DIR.iterdir() if SESSIONS_DIR.is_dir() else []:
            analysis_id = session_directory.name
            if (
                not session_directory.is_dir()
                or len(analysis_id) != ANALYSIS_ID_LENGTH
                or any(character not in "0123456789abcdef" for character in analysis_id)
            ):
                continue
            manifest_path = session_directory / "analysis.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping invalid legacy manifest for analysis %s.", analysis_id)
                invalid += 1
                continue
            if (
                not isinstance(manifest, dict)
                or manifest.get("analysis_id") != analysis_id
                or manifest.get("status") != "completed"
                or not isinstance(manifest.get("video_filename"), str)
                or not manifest["video_filename"].strip()
                or not isinstance(manifest.get("class_counts", {}), dict)
                or not isinstance(manifest.get("detections", []), list)
                or not isinstance(manifest.get("tracking_events", []), list)
            ):
                logger.warning("Skipping incomplete legacy manifest for analysis %s.", analysis_id)
                invalid += 1
                continue
            if session.get(AnalysisRecord, analysis_id) is not None:
                skipped += 1
                continue
            analysis_dir = COUNTING_OUTPUT_DIR / f"analysis_{analysis_id}"
            output_path = _path_from_public_url(
                manifest.get("output_video_url"), analysis_id
            )
            if output_path is None:
                output_path = COUNTING_OUTPUT_DIR / f"vehicle_count_{analysis_id}.mp4"
            artifacts = _manifest_artifacts(manifest, analysis_id)
            artifacts.update(
                {
                    "analysis_dir": _relative_path(analysis_dir),
                    "session_dir": _relative_path(session_directory),
                    "annotated_video": _relative_path(output_path),
                }
            )
            if not output_path.is_file():
                logger.warning("Legacy analysis %s has no annotated video file.", analysis_id)
            if not dry_run:
                session.add(
                    AnalysisRecord(
                        id=analysis_id,
                        filename=Path(manifest["video_filename"]).name[:255],
                        created_at=_manifest_datetime(manifest, manifest_path),
                        completed_at=_manifest_datetime(manifest, manifest_path),
                        status="completed",
                        duration_seconds=manifest.get("duration_seconds"),
                        line_position_percent=manifest.get("line_position_percent"),
                        total_count=manifest.get("total_line_crossings"),
                        class_counts=_json_value(manifest.get("class_counts") or {}),
                        tracking_events=_json_value(manifest.get("tracking_events") or []),
                        anpr_records=_json_value(manifest.get("detections") or []),
                        search_results=_json_value(manifest.get("search_results") or []),
                        artifact_paths=artifacts,
                        manifest=_json_value(manifest),
                        export_metadata={"available": ["csv", "excel", "pdf", "zip"]},
                    )
                )
            imported += 1
        if not dry_run:
            session.commit()
    return {"imported": imported, "skipped": skipped, "invalid": invalid}
