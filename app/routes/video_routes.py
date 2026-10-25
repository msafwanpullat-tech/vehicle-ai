import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from app.services.video_processor import VehicleDetectorService

router = APIRouter(prefix="/api", tags=["video"])
logger = logging.getLogger("vehicle_ai.routes")

try:
    detector_service = VehicleDetectorService(model_path="models/yolo11n.pt")
except Exception as e:
    logger.error(f"Failed to initialize VehicleDetectorService: {e}")
    detector_service = None

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@router.get("/status")
def get_system_status():
    """Returns operational status of vehicle analysis & license plate OCR engine."""
    return {
        "status": "online",
        "model_loaded": detector_service is not None,
        "supported_classes": ["car", "motorcycle", "bus", "truck"],
        "ocr_engine": "EasyOCR / LicensePlateDetector",
        "device": "CPU"
    }

@router.post("/process-video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    line_ratio: float = Form(0.5)
):
    """
    Endpoint to upload a traffic video, run ByteTrack vehicle tracking, line counting,
    and license plate OCR recognition.
    """
    if not detector_service:
        raise HTTPException(status_code=500, detail="Vehicle detector service is not initialized.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file selected for upload.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    unique_id = uuid.uuid4().hex[:10]
    upload_filename = f"upload_{unique_id}{ext}"
    output_filename = f"output_{unique_id}.mp4"

    upload_path = os.path.join("uploads", upload_filename)
    output_path = os.path.join("outputs", output_filename)

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video upload: {str(e)}")

    try:
        result = detector_service.tracker.track_video(
            input_path=upload_path,
            output_path=output_path,
            line_ratio=line_ratio
        )
        
        result["original_video_url"] = f"/uploads/{upload_filename}"
        result["output_video_url"] = f"/outputs/{output_filename}"
        result["filename"] = file.filename
        
        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Error processing video")
        raise HTTPException(status_code=500, detail=f"Error during video AI processing: {str(e)}")

@router.get("/plates")
def get_all_recognized_plates():
    """Returns all recognized vehicle license plate records from recent processing session."""
    if not detector_service:
        raise HTTPException(status_code=500, detail="Service uninitialized.")
    
    records = detector_service.tracker.plate_registry.get_all_records()
    return JSONResponse(content={"count": len(records), "plates": records})

@router.get("/search")
def search_registration_endpoint(q: str = Query(..., description="Registration number search string")):
    """
    Search endpoint to find vehicles by registration number.
    Returns matching vehicle ID, type, registration number, confidence, timestamp, and frame URL.
    """
    if not detector_service:
        raise HTTPException(status_code=500, detail="Service uninitialized.")

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Search query string cannot be empty.")

    matches = detector_service.tracker.plate_registry.search_registration(q)

    if not matches:
        return JSONResponse(content={
            "query": q,
            "count": 0,
            "results": [],
            "message": "No matching registration number was found."
        })

    return JSONResponse(content={
        "query": q,
        "count": len(matches),
        "results": matches,
        "message": f"Found {len(matches)} matching vehicle record(s)."
    })
