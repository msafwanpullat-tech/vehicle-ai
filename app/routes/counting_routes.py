import asyncio
import json
import logging
import math
import os
import re
import shutil
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import PROJECT_ROOT, get_db
from app.models.analysis_history import AnalysisRecord
from app.services.analysis_history import (
    complete_analysis,
    create_analysis,
    delete_analysis_record,
    fail_analysis,
    get_analysis,
    get_analysis_manifest,
    list_analyses,
    save_manifest,
)
from app.services.anpr_service import PlateRecognitionProcessor
from app.services.export_service import (
    generate_asset_zip,
    generate_exports,
    generate_results_zip,
)
from app.services.registration_search import is_valid_registration, search_detections
from count_vehicles import DEFAULT_MODEL, InvalidVideoError, process_video

router = APIRouter(prefix="/api/counting", tags=["vehicle-counting"])
logger = logging.getLogger("vehicle_ai.counting_routes")

UPLOAD_DIR = PROJECT_ROOT / "uploads" / "counting"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "counting"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
OUTPUT_EXTENSION = ".mp4" if os.name == "nt" else ".webm"
OUTPUT_MEDIA_TYPE = "video/mp4" if OUTPUT_EXTENSION == ".mp4" else "video/webm"
processing_lock = asyncio.Lock()
manifest_lock = threading.Lock()


class RegistrationSearchRequest(BaseModel):
    registration_number: str = Field(min_length=4, max_length=32)


@router.get("/status")
def get_counting_status() -> dict[str, bool | str]:
    return {
        "status": "ready" if DEFAULT_MODEL.is_file() else "model_unavailable",
        "model_available": DEFAULT_MODEL.is_file(),
        "device": "CPU",
        "tracker": "ByteTrack",
    }


def _video_duration(path: Path) -> float | None:
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(frame_count) or not math.isfinite(fps):
            return None
        if frame_count <= 0 or fps <= 0:
            return None
        return round(frame_count / fps, 3)
    finally:
        capture.release()


def _record_failure(analysis_id: str, message: str) -> None:
    try:
        fail_analysis(analysis_id, message)
    except Exception:
        logger.exception("Could not persist failure status for analysis %s.", analysis_id)


@router.post("/process")
async def count_uploaded_video(
    file: UploadFile = File(...),
    line_position_percent: float = Form(default=55.6, ge=0, le=100),
    debug_anpr: bool = Form(default=False),
) -> dict[str, object]:
    original_filename = file.filename
    if not original_filename:
        await file.close()
        raise HTTPException(status_code=400, detail="Choose a video file to upload.")

    extension = Path(original_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Unsupported video format. Upload an MP4, AVI, MOV, MKV, or WEBM file.",
        )
    if not DEFAULT_MODEL.is_file():
        await file.close()
        raise HTTPException(
            status_code=503,
            detail="The vehicle detection model is unavailable. Please contact the administrator.",
        )

    job_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{job_id}{extension}"
    analysis_dir = OUTPUT_DIR / f"analysis_{job_id}"
    output_path = analysis_dir / f"vehicle_count_{job_id}{OUTPUT_EXTENSION}"
    session_dir = PROJECT_ROOT / ".analysis_sessions" / job_id
    upload_size = 0
    analysis_completed = False
    analysis_record_created = False

    try:
        create_analysis(
            job_id,
            Path(original_filename).name,
            line_position_percent,
            analysis_dir,
            session_dir,
            output_path.name,
        )
        analysis_record_created = True
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with upload_path.open("wb") as saved_video:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                upload_size += len(chunk)
                if upload_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The uploaded video exceeds the 500 MB size limit.",
                    )
                saved_video.write(chunk)
            saved_video.flush()
            os.fsync(saved_video.fileno())
        if upload_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        duration_seconds = _video_duration(upload_path)

        async with processing_lock:
            anpr_debug_dir = PROJECT_ROOT / ".anpr_debug" / job_id
            anpr_processor = PlateRecognitionProcessor(
                debug_dir=anpr_debug_dir if debug_anpr else None,
                analysis_dir=analysis_dir,
                session_dir=session_dir,
            )
            total_count, class_counts = await asyncio.to_thread(
                process_video,
                DEFAULT_MODEL,
                upload_path,
                output_path,
                line_position_percent,
                browser_compatible=True,
                anpr_processor=anpr_processor,
            )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        source_path = analysis_dir / f"source{extension}"
        shutil.copy2(upload_path, source_path)
        output_video_url = f"/outputs/counting/analysis_{job_id}/{output_path.name}"
        source_video_url = (
            f"/outputs/counting/analysis_{job_id}/{source_path.name}"
        )
        anpr_processor.summary = {
            "analysis_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "video_filename": Path(original_filename).name,
            "line_position_percent": line_position_percent,
            "total_line_crossings": total_count,
            "class_counts": class_counts,
            "total_tracking_events": len(anpr_processor.tracking_events),
            "anpr_status": anpr_processor.status,
            "status": "completed",
            "duration_seconds": duration_seconds,
            "processing_settings": {
                "tracker": "ByteTrack",
                "device": "CPU",
                "model": "Custom YOLO11",
            },
        }
        anpr_processor.save_analysis_manifest(output_video_url, source_video_url)
        manifest_path = session_dir / "analysis.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete_analysis(job_id, manifest, duration_seconds)
        analysis_completed = True
        response: dict[str, object] = {
            "analysis_id": job_id,
            "filename": Path(original_filename).name,
            "status": "completed",
            "created_at": anpr_processor.summary["created_at"],
            "duration_seconds": duration_seconds,
            "total_count": total_count,
            "class_counts": class_counts,
            "line_position_percent": line_position_percent,
            "plate_results": anpr_processor.get_records(),
            "anpr_status": anpr_processor.status,
            "output_video_url": output_video_url,
            "source_video_url": source_video_url,
            "download_url": f"/api/counting/{job_id}/download",
            "search_url": f"/api/counting/{job_id}/search",
            "export_urls": {
                "csv": f"/api/counting/{job_id}/export/csv",
                "excel": f"/api/counting/{job_id}/export/excel",
                "pdf": f"/api/counting/{job_id}/export/pdf",
                "zip": f"/api/counting/{job_id}/export/all.zip",
                "vehicles": f"/api/counting/{job_id}/export/assets/vehicles",
                "plates": f"/api/counting/{job_id}/export/assets/plates",
            },
        }
        if debug_anpr:
            response["debug_candidates_saved"] = anpr_processor.ocr_attempts
            logger.info("Saved optional ANPR debug artifacts under %s", anpr_debug_dir)
        return response
    except HTTPException:
        if analysis_record_created:
            _record_failure(job_id, str(
                "The upload exceeded the size limit."
                if upload_size > MAX_UPLOAD_BYTES
                else "The video could not be processed."
            ))
        raise
    except InvalidVideoError:
        logger.info("Rejected an uploaded file that was not a readable video.")
        if analysis_record_created:
            _record_failure(
                job_id,
                "The video could not be decoded. It may be damaged or use an unsupported codec.",
            )
        raise HTTPException(
            status_code=422,
            detail=(
                "The video could not be decoded. It may be damaged or use an unsupported "
                "codec; try re-encoding it as H.264 MP4 or MJPEG AVI."
            ),
        ) from None
    except FileNotFoundError:
        logger.exception("The configured vehicle detection model became unavailable.")
        if analysis_record_created:
            _record_failure(
                job_id,
                "The vehicle detection model is unavailable.",
            )
        raise HTTPException(
            status_code=503,
            detail="The vehicle detection model is unavailable. Please contact the administrator.",
        ) from None
    except Exception:
        logger.exception("Vehicle counting failed while processing an uploaded video.")
        if analysis_record_created:
            _record_failure(
                job_id,
                "Video processing failed. Please try another video or contact the administrator.",
            )
        raise HTTPException(
            status_code=500,
            detail="Video processing failed. Please try another video or contact the administrator.",
        ) from None
    finally:
        await file.close()
        for temporary_path in (upload_path, None if analysis_completed else output_path):
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove a temporary counting file.")
        if not analysis_completed:
            for temporary_directory in (analysis_dir, session_dir):
                try:
                    shutil.rmtree(temporary_directory)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Could not remove incomplete analysis artifacts.")


@router.post("/{job_id}/search")
def search_vehicle_registration(
    job_id: str, request: RegistrationSearchRequest
) -> dict[str, object]:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Video analysis not found.")
    normalized_query = request.registration_number.strip().upper()
    if not is_valid_registration(normalized_query):
        raise HTTPException(
            status_code=422,
            detail="Enter a valid Indian registration number, such as KL07AB1234.",
        )

    manifest_path = PROJECT_ROOT / ".analysis_sessions" / job_id / "analysis.json"
    manifest = get_analysis_manifest(job_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Video analysis not found.")
    try:
        result = search_detections(normalized_query, manifest.get("detections", []))
        safe_manifest = _safe_analysis_manifest(job_id, manifest)
        output_video_url = safe_manifest.get("output_video_url")
        source_video_url = safe_manifest.get("source_video_url")
        stored_search = {
            "query": result["query"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": (
                "Vehicle registration number not found in this video."
                if result["not_found"]
                else (
                    "Exact confirmed registration match found."
                    if result["exact_matches"]
                    else "No confirmed exact match. Similar OCR candidate(s) are uncertain."
                )
            ),
            "not_found": result["not_found"],
            "exact_matches": result["exact_matches"],
            "uncertain_candidates": result["uncertain_candidates"],
        }
        with manifest_lock:
            manifest.setdefault("search_results", []).append(stored_search)
            if manifest_path.is_file():
                temporary_path = manifest_path.with_suffix(".json.tmp")
                temporary_path.write_text(
                    json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
                )
                temporary_path.replace(manifest_path)
        save_manifest(job_id, manifest)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.exception("Could not read saved vehicle search data for analysis %s.", job_id)
        raise HTTPException(
            status_code=500,
            detail="Vehicle search data is unavailable for this analysis.",
        ) from None

    result["analysis_id"] = job_id
    result["output_video_url"] = output_video_url
    result["source_video_url"] = source_video_url
    if result["not_found"]:
        result["message"] = "Vehicle registration number not found in this video."
    elif result["exact_matches"]:
        result["message"] = "Exact confirmed registration match found."
    else:
        result["message"] = (
            "No confirmed exact match. Similar OCR candidate(s) are uncertain; "
            "verify the plate image before drawing a conclusion."
        )
    return result


def _safe_analysis_manifest(analysis_id: str, manifest: dict[str, object]) -> dict[str, object]:
    safe_manifest = dict(manifest)
    analysis_dir = (OUTPUT_DIR / f"analysis_{analysis_id}").resolve()

    def safe_asset_url(value: object) -> str | None:
        if not isinstance(value, str) or not value.startswith(
            f"/outputs/counting/analysis_{analysis_id}/"
        ):
            return None
        filename = Path(value.split("?", 1)[0]).name
        candidate = (analysis_dir / filename).resolve()
        try:
            candidate.relative_to(analysis_dir)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return f"/outputs/counting/analysis_{analysis_id}/{filename}"

    detections = manifest.get("detections", [])
    safe_detections: list[dict[str, object]] = []
    if isinstance(detections, list):
        for item in detections:
            if not isinstance(item, dict):
                continue
            safe_item = dict(item)
            safe_item["vehicle_image_url"] = safe_asset_url(item.get("vehicle_image_url"))
            safe_item["plate_image_url"] = safe_asset_url(item.get("plate_image_url"))
            safe_detections.append(safe_item)
    safe_manifest["detections"] = safe_detections

    def safe_video_path(value: object, legacy_name: str) -> Path | None:
        if not isinstance(value, str) or not value.startswith("/outputs/counting/"):
            return None
        relative = value.split("?", 1)[0].removeprefix("/outputs/counting/")
        candidate = (OUTPUT_DIR / relative).resolve()
        try:
            candidate.relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            return None
        allowed_parent = candidate.parent == analysis_dir or (
            candidate.parent == OUTPUT_DIR.resolve()
            and candidate.name == legacy_name
        )
        return candidate if allowed_parent and candidate.is_file() else None

    output_path = safe_video_path(
        manifest.get("output_video_url"),
        f"vehicle_count_{analysis_id}{OUTPUT_EXTENSION}",
    )
    source_path = safe_video_path(
        manifest.get("source_video_url"), f"source{Path(str(manifest.get('source_video_url', ''))).suffix}"
    )
    safe_manifest["output_video_url"] = (
        f"/outputs/counting/{output_path.relative_to(OUTPUT_DIR).as_posix()}"
        if output_path is not None
        else None
    )
    safe_manifest["source_video_url"] = (
        f"/outputs/counting/{source_path.relative_to(OUTPUT_DIR).as_posix()}"
        if source_path is not None
        else None
    )
    return safe_manifest


@router.get("/history")
def get_analysis_history(
    search: str = Query(default="", max_length=255),
    status: str = Query(default="all", pattern="^(all|processing|completed|failed)$"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    sort: str = Query(default="newest", pattern="^(newest|oldest)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="Start date must be on or before end date.")
    records, total = list_analyses(
        search.strip(), status, date_from, date_to, sort, page, page_size, db=db
    )
    return {
        "items": records,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }


@router.get("/{job_id}/details")
def get_analysis_details(job_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Video analysis not found.")
    record = db.get(AnalysisRecord, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video analysis not found.")
    result: dict[str, object] = {
        "analysis_id": record.id,
        "filename": record.filename,
        "status": record.status,
        "created_at": record.created_at,
        "duration_seconds": record.duration_seconds,
        "line_position_percent": record.line_position_percent,
        "total_count": record.total_count,
        "class_counts": record.class_counts or {},
        "failure_message": record.failure_message,
    }
    if record.status == "completed" and isinstance(record.manifest, dict):
        manifest = _safe_analysis_manifest(record.id, record.manifest)
        result.update(
            {
                "total_count": manifest.get("total_line_crossings", record.total_count),
                "class_counts": manifest.get("class_counts", record.class_counts or {}),
                "plate_results": manifest.get("detections", []),
                "tracking_events": manifest.get("tracking_events", []),
                "search_results": manifest.get("search_results", []),
                "processing_settings": manifest.get("processing_settings", {}),
                "anpr_status": manifest.get("anpr_status", "unavailable"),
                "output_video_url": manifest.get("output_video_url"),
                "source_video_url": manifest.get("source_video_url"),
                "download_url": f"/api/counting/{record.id}/download",
                "search_url": f"/api/counting/{record.id}/search",
                "export_urls": {
                    "csv": f"/api/counting/{record.id}/export/csv",
                    "excel": f"/api/counting/{record.id}/export/excel",
                    "pdf": f"/api/counting/{record.id}/export/pdf",
                    "zip": f"/api/counting/{record.id}/export/all.zip",
                    "vehicles": f"/api/counting/{record.id}/export/assets/vehicles",
                    "plates": f"/api/counting/{record.id}/export/assets/plates",
                },
            }
        )
    return result


@router.delete("/{job_id}/history")
def delete_analysis_history(job_id: str) -> dict[str, object]:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Video analysis not found.")
    if get_analysis(job_id) is None:
        raise HTTPException(status_code=404, detail="Video analysis not found.")

    trash_root = PROJECT_ROOT / "data" / ".analysis_trash" / f"{job_id}_{uuid.uuid4().hex}"
    targets = [
        OUTPUT_DIR / f"analysis_{job_id}",
        PROJECT_ROOT / ".analysis_sessions" / job_id,
        OUTPUT_DIR / f"vehicle_count_{job_id}.mp4",
        OUTPUT_DIR / f"vehicle_count_{job_id}.webm",
        PROJECT_ROOT / ".anpr_debug" / job_id,
    ]
    staged: list[tuple[Path, Path]] = []
    try:
        for index, target in enumerate(targets):
            resolved = target.resolve()
            try:
                resolved.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                raise HTTPException(status_code=404, detail="Video analysis not found.") from None
            if not target.exists():
                continue
            trash_root.mkdir(parents=True, exist_ok=True)
            staged_path = trash_root / f"{index}_{target.name}"
            os.replace(target, staged_path)
            staged.append((staged_path, target))
        if not delete_analysis_record(job_id):
            raise RuntimeError("Analysis history record disappeared during deletion.")
    except Exception as error:
        for staged_path, original_path in reversed(staged):
            try:
                original_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, original_path)
            except OSError:
                logger.exception("Could not restore an analysis artifact after delete failure.")
        if isinstance(error, HTTPException):
            raise
        logger.exception("Could not safely delete analysis %s.", job_id)
        raise HTTPException(
            status_code=500, detail="Could not delete this analysis safely. Please try again."
        ) from None

    cleanup_warning = None
    if trash_root.exists():
        try:
            shutil.rmtree(trash_root)
        except OSError:
            logger.exception("Analysis %s was deleted, but its staged files remain.", job_id)
            cleanup_warning = "The record was removed, but some files could not be cleared."
    return {"deleted": True, "cleanup_warning": cleanup_warning}


def _analysis_paths(job_id: str) -> tuple[Path, Path, Path]:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Analysis export not found.")
    record = get_analysis(job_id)
    if record is None or record.status != "completed" or not isinstance(record.manifest, dict):
        raise HTTPException(status_code=404, detail="Analysis export not found.")
    manifest_path = PROJECT_ROOT / ".analysis_sessions" / job_id / "analysis.json"
    analysis_dir = (PROJECT_ROOT / "outputs" / "counting" / f"analysis_{job_id}").resolve()
    try:
        analysis_dir.relative_to((PROJECT_ROOT / "outputs" / "counting").resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Analysis export not found.") from None
    output_value = record.manifest.get("output_video_url")
    if not isinstance(output_value, str) or not output_value.startswith("/outputs/counting/"):
        raise HTTPException(status_code=404, detail="Analysis video is unavailable.")
    relative_video = output_value.split("?", 1)[0].removeprefix("/outputs/counting/")
    output_path = (OUTPUT_DIR / relative_video).resolve()
    try:
        output_path.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Analysis video is unavailable.") from None
    allowed_legacy_output = (
        output_path.parent == OUTPUT_DIR.resolve()
        and output_path.name
        in {f"vehicle_count_{job_id}.mp4", f"vehicle_count_{job_id}.webm"}
    )
    if output_path.parent != analysis_dir and not allowed_legacy_output:
        raise HTTPException(status_code=404, detail="Analysis video is unavailable.")
    if not analysis_dir.is_dir():
        raise HTTPException(status_code=404, detail="Analysis files are unavailable.")
    manifest_path = analysis_dir / f".export-manifest-{uuid.uuid4().hex}.json"
    manifest_path.write_text(
        json.dumps(record.manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return manifest_path, analysis_dir, output_path


def _read_export_manifest(manifest_path: Path) -> dict[str, object]:
    try:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read saved analysis export data.")
        raise HTTPException(
            status_code=500, detail="Saved analysis data is unavailable."
        ) from None
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500, detail="Saved analysis data is unavailable."
        )
    return result


@router.get("/{job_id}/export/all.zip")
def download_all_analysis_results(job_id: str) -> FileResponse:
    manifest_path, analysis_dir, output_path = _analysis_paths(job_id)
    manifest = _read_export_manifest(manifest_path)
    path = analysis_dir / "all_results.zip"
    try:
        generate_results_zip(
            job_id, manifest_path, manifest, analysis_dir, output_path, path
        )
    except Exception:
        logger.exception("Could not create complete export archive for analysis %s.", job_id)
        raise HTTPException(
            status_code=500, detail="Could not prepare the complete analysis archive."
        ) from None
    finally:
        if manifest_path.name.startswith(".export-manifest-"):
            manifest_path.unlink(missing_ok=True)
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"vehicle_analysis_{job_id}.zip",
    )


@router.get("/{job_id}/export/{export_format}")
def download_analysis_export(job_id: str, export_format: str) -> FileResponse:
    if export_format not in {"csv", "excel", "pdf"}:
        raise HTTPException(status_code=404, detail="Analysis export not found.")
    manifest_path, analysis_dir, output_path = _analysis_paths(job_id)
    manifest = _read_export_manifest(manifest_path)
    try:
        exports = generate_exports(job_id, manifest, analysis_dir, output_path)
    except Exception:
        logger.exception("Could not generate %s export for analysis %s.", export_format, job_id)
        raise HTTPException(
            status_code=500, detail="Could not generate this analysis export."
        ) from None
    finally:
        if manifest_path.name.startswith(".export-manifest-"):
            manifest_path.unlink(missing_ok=True)
    path = exports[export_format]
    media_types = {
        "csv": "text/csv; charset=utf-8",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
    }
    names = {
        "csv": f"vehicle_analysis_{job_id}.csv",
        "excel": f"vehicle_analysis_{job_id}.xlsx",
        "pdf": f"vehicle_analysis_{job_id}.pdf",
    }
    return FileResponse(path, media_type=media_types[export_format], filename=names[export_format])


@router.get("/{job_id}/export/assets/{asset_kind}")
def download_analysis_assets(job_id: str, asset_kind: str) -> FileResponse:
    if asset_kind not in {"vehicles", "plates"}:
        raise HTTPException(status_code=404, detail="Analysis screenshots not found.")
    manifest_path, analysis_dir, _ = _analysis_paths(job_id)
    path = analysis_dir / f"{asset_kind}_screenshots.zip"
    try:
        generate_asset_zip(analysis_dir, asset_kind, path)
    except OSError:
        logger.exception("Could not create %s screenshot archive.", asset_kind)
        raise HTTPException(
            status_code=500, detail="Could not prepare analysis screenshots."
        ) from None
    finally:
        if manifest_path.name.startswith(".export-manifest-"):
            manifest_path.unlink(missing_ok=True)
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"vehicle_{asset_kind}_{job_id}.zip",
    )


@router.get("/{job_id}/download")
def download_counted_video(job_id: str) -> FileResponse:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Processed video not found.")
    record = get_analysis(job_id)
    if record is None or record.status != "completed" or not isinstance(record.manifest, dict):
        raise HTTPException(status_code=404, detail="Processed video not found.")
    output_url = record.manifest.get("output_video_url")
    if not isinstance(output_url, str) or not output_url.startswith("/outputs/counting/"):
        raise HTTPException(status_code=404, detail="Processed video not found.")
    relative_video = output_url.split("?", 1)[0].removeprefix("/outputs/counting/")
    output_path = (OUTPUT_DIR / relative_video).resolve()
    try:
        output_path.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Processed video not found.") from None
    expected_analysis_dir = (OUTPUT_DIR / f"analysis_{job_id}").resolve()
    if (
        output_path.parent not in {expected_analysis_dir, OUTPUT_DIR.resolve()}
        or output_path.parent == OUTPUT_DIR.resolve()
        and output_path.name not in {
            f"vehicle_count_{job_id}.mp4",
            f"vehicle_count_{job_id}.webm",
        }
        or not output_path.is_file()
    ):
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(
        path=output_path,
        media_type=OUTPUT_MEDIA_TYPE,
        filename=output_path.name,
    )
