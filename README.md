# AI-Based Vehicle Detection, Classification and Counting System using YOLO

An academic Computer Vision mini project built with Python, FastAPI, Ultralytics YOLO11, ByteTrack, and License Plate OCR.

---

## Project Phase Progress

- **Phase 1**: Traffic video upload, YOLO11 vehicle detection, bounding boxes, web dashboard, and API endpoints.
- **Phase 2**: Multi-Object Vehicle Tracking using ByteTrack with persistent tracking IDs (`Car | ID: 7`, `Bus | ID: 4`, etc.).
- **Phase 3**: Virtual Line Crossing Vehicle Counter with directional tracking (`IN` / `OUT`) and duplicate suppression.
- **Phase 4**: License Plate Detection, Preprocessing & OCR Engine, Vehicle-to-Plate Association, and Registration Number Search.

---

## How Phase 4 License Plate Detection & OCR Works

### 1. License Plate Detection Module ([`plate_detector.py`](app/services/plate_detector.py))
- Focuses analysis on the lower 65% of detected vehicle ROIs.
- Applies morphological Blackhat filtering and Sobel vertical edge detection to isolate character density.
- Filters contours by rectangular aspect ratio ($2.0 \le \text{Aspect Ratio} \le 6.0$).
- Crops candidate license plate image regions for OCR recognition.

### 2. Preprocessing & OCR Pipeline ([`ocr_service.py`](app/services/ocr_service.py))
- **Preprocessing**:
  1. Resizes crop to standard height (80px).
  2. Grayscale conversion.
  3. CLAHE (Contrast Limited Adaptive Histogram Equalization).
  4. Bilateral filtering for edge-preserving noise reduction.
  5. Otsu thresholding.
- **Text Normalization**:
  - Converts text to uppercase.
  - Removes spaces, hyphens, and non-alphanumeric characters.
  - Formats registration strings (e.g. `KL07AB1234`).
- **Confidence Scoring**: Reports estimated OCR confidence percentage (e.g. `91%`).

### 3. Vehicle-to-Plate Association & Throttling ([`plate_registry.py`](app/services/plate_registry.py))
- Maintains persistent association: `Vehicle ID -> Registration Number`.
- Throttles OCR execution to run every 15 frames per vehicle ID, saving CPU overhead and preventing repeated OCR calls.
- Retains the highest-confidence recognized registration number per vehicle ID.

### 4. Registration Number Search (`GET /api/search?q=...`)
- Allows users to query recognized registration numbers (exact or partial match).
- Returns vehicle ID, vehicle type, recognized plate, confidence %, timestamp, and relevant video frame image.

---

## Modular Directory Structure

```
vehicle-ai/
│
├── .venv/                   # Python virtual environment
├── app/
│   ├── main.py              # FastAPI application entry point & static file mounts
│   ├── routes/
│   │   ├── video_routes.py  # API routes for upload, line counting, OCR & search
│   │   └── counting_routes.py # Upload and counting dashboard API endpoints
│   ├── services/
│   │   ├── counter.py         # Virtual Line Crossing Counter engine
│   │   ├── plate_detector.py  # License plate region detector & cropper
│   │   ├── ocr_service.py     # Image preprocessing & OCR engine
│   │   ├── plate_registry.py  # Vehicle-to-plate association & search manager
│   │   ├── vehicle_tracker.py # ByteTrack tracking & OCR pipeline integration
│   │   └── video_processor.py # Service wrapper
│   ├── models/
│   ├── utils/
│   └── templates/
│       ├── index.html       # Existing analysis dashboard template
│       └── counting_dashboard.html
│
├── static/
│   ├── css/
│   │   ├── style.css        # Existing dashboard styling
│   │   └── counting-dashboard.css
│   └── js/
│       ├── app.js           # Existing dashboard interactions
│       └── counting-dashboard.js
│
├── models/
│   └── yolo11n.pt           # Lightweight YOLO11 model weights
├── videos/                  # Sample & uploaded source videos
├── uploads/                 # Video upload storage
├── outputs/                 # AI processed output video storage
│   ├── plates/              # License plate crop snapshots
│   └── frames/              # Matched video frame snapshots
├── tests/
│   ├── create_sample_video.py  # Test video generator
│   ├── test_detection.py       # Standalone detection test
│   ├── test_tracking.py        # ByteTrack & line counting integration test
│   ├── test_line_counter.py    # Line counter unit test
│   └── test_ocr_plate.py       # OCR & Plate Registry unit test
├── count_vehicles.py        # Standalone ByteTrack virtual-line vehicle counter
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation & run guide
```

---

## Run Standalone Vehicle Counting

From the project root, run the counter with the custom model and sample traffic video:

```powershell
.\.venv\Scripts\python.exe count_vehicles.py
```

The processed video is saved to `runs\vehicle_counting_output.mp4`, and total and
class-wise counts are printed in the terminal. Adjust the horizontal counting line
with `--line-position` (percentage of frame height, from `0` to `100`), and use
`--debug` to print each tracked vehicle center and crossing status:

```powershell
.\.venv\Scripts\python.exe count_vehicles.py --line-position 55.6 --debug
```

The legacy `--line-ratio` option (fraction from `0.0` to `1.0`) remains available.
Class-wise totals use the majority class observed for each tracking ID to reduce
single-frame class-label flicker. The total is the sum of class counts, which are
incremented only when a tracked center crosses the line. Each tracking ID is counted
once; if tracking is interrupted and the same real-world vehicle gets a new ID, it
can be counted again. Counts therefore represent unique tracked-ID line-crossing
events, not guaranteed distinct real-world vehicles. Choose a line position that
spans the traffic lanes in the input video.

---

## Vehicle Counting Web Dashboard

Install project dependencies if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the FastAPI dashboard from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The dashboard accepts MP4,
AVI, MOV, MKV, and WEBM uploads up to 500 MB, and allows the virtual counting line
to be positioned from 10% to 90% of frame height. The temporary upload is removed
after the source video is copied into its analysis-specific folder; generated videos
remain available under `outputs\counting`. Dashboard outputs use browser-compatible VP8/WebM encoding; the
standalone script continues to use MP4 by default. Dashboard outputs use H.264/MP4
through Windows Media Foundation on Windows, and VP8/WebM on other platforms.

The counting endpoint is `POST /api/counting/process` (multipart fields `file` and
`line_position_percent`). `GET /api/counting/status` reports custom-model availability.

### Local number-plate recognition

Dashboard processing also checks tracked vehicle crops with the project's OpenCV
contour-based plate candidate detector and EasyOCR on the CPU. This is a heuristic
candidate detector, not a separately trained plate-detection model. Plate results
are linked to the vehicle track ID and shown with OCR confidence. Text is marked
recognized only when it matches an Indian registration-number pattern and reaches
70% OCR confidence; other candidates are labeled uncertain, unreadable, or OCR
unavailable. Candidate OCR is limited to once per track every 15 frames, and
candidate boxes/text are overlaid on the processed output.

EasyOCR's English recognition weights are downloaded from its configured model
source on first use and cached locally; uploaded videos are processed locally and
are not sent to an external service. The environment also needs the `lazy-loader`
dependency used by the already-listed scikit-image/EasyOCR packages:

```powershell
.\.venv\Scripts\python.exe -m pip install lazy-loader
```

### PostgreSQL analysis history

History is stored in PostgreSQL using SQLAlchemy and the psycopg driver. Video and
image assets remain in the per-analysis folders under `outputs\counting`; no media
files are stored in the database. Migrations are explicit and are not run during
FastAPI startup.

1. Install the new backend dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` only if `.env` does not already exist, then edit
   `DATABASE_URL` locally using the PostgreSQL credentials for your existing
   database. Do not paste credentials into chat or commit `.env`.

   ```powershell
   if (-not (Test-Path .env)) { Copy-Item .env.example .env }
   ```

   The accepted URL format is
   `postgresql+psycopg://USERNAME:PASSWORD@localhost:5432/DATABASE_NAME`.
   The values shown here are placeholders; use the database name and account
   already provisioned on your machine. Environment variables override `.env`.

3. Inspect the existing PostgreSQL tables before migration, replacing the two
   placeholders locally with the already-provisioned account and database name.
   `psql` prompts for a password if required. The migration refuses to create
   `analysis_records` if that table already exists, so review any conflict rather
   than dropping or resetting data. Then apply the migration:

   ```powershell
   psql -h localhost -U YOUR_EXISTING_USER -d YOUR_EXISTING_DATABASE -c "\dt"
   .\.venv\Scripts\alembic.exe current
   .\.venv\Scripts\alembic.exe upgrade head
   ```

4. Existing JSON manifests are not imported automatically. Preview valid,
   non-duplicate legacy records first, then explicitly commit the import:

   ```powershell
   .\.venv\Scripts\python.exe scripts\import_analysis_history.py
   .\.venv\Scripts\python.exe scripts\import_analysis_history.py --apply
   ```

   The importer is repeatable, preserves IDs and artifact references, and logs
   missing files without deleting them.

5. Start the application after configuring PostgreSQL and applying the migration:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

FastAPI verifies PostgreSQL connectivity at startup but never creates tables or
runs migrations automatically. Analysis history can be searched, viewed without
rerunning inference, exported, and deleted per analysis.

The sample traffic video may not contain plates large or clear enough to recognize.
In that case the dashboard reports no plate candidates or unreadable results rather
than presenting low-confidence text as a confirmed registration.

For temporary, local crop/OCR diagnostics, include `debug_anpr=true` in the
`/api/counting/process` multipart form. Candidate frame, vehicle crop, padded plate
crop, OCR input variants, and JSON OCR results are saved under
`.anpr_debug\<job-id>` (not exposed by the static output route). This is off by
default because diagnostic crops may contain identifiable registration
information; remove the debug folder when done.

After a video is processed, use **Search Vehicle by Registration Number** in the
dashboard. Search is scoped to that analysis and accepts Indian registration
numbers with spaces or hyphens. Confirmed results require the normal ANPR
registration-format and confidence checks; near OCR strings are shown separately
as uncertain candidates for visual review. The original upload, processed video,
vehicle crops, and plate crops are saved in unique per-analysis output folders.
The search endpoint is `POST /api/counting/{analysis_id}/search` with a JSON
`registration_number` field.

### Per-analysis downloads and reports

The results section offers CSV, Excel, PDF, complete ZIP, vehicle-crop, and
number-plate-crop downloads, in addition to the annotated-video download. Every
export is built from the saved manifest and crops for that analysis; exporting
does not rerun video inference. Search attempts made for an analysis are included
in later exports, including explicit no-match results and uncertain OCR candidates.

| Method | Endpoint | Download |
|--------|----------|----------|
| `GET` | `/api/counting/{analysis_id}/export/csv` | Combined summary, class counts, tracking, OCR, and search CSV |
| `GET` | `/api/counting/{analysis_id}/export/excel` | Workbook with Summary, Class Counts, Vehicle Events, Number Plates, and Vehicle Search Results sheets |
| `GET` | `/api/counting/{analysis_id}/export/pdf` | Analysis report with distribution chart, event/OCR/search summaries, selected crops, and limitations |
| `GET` | `/api/counting/{analysis_id}/export/all.zip` | CSV, workbook, PDF, manifest, available crops, and annotated video if at most 100 MiB |
| `GET` | `/api/counting/{analysis_id}/export/assets/vehicles` | ZIP of saved vehicle crops |
| `GET` | `/api/counting/{analysis_id}/export/assets/plates` | ZIP of saved plate crops |

The standalone annotated-video endpoint remains available at
`/api/counting/{analysis_id}/download`. The complete ZIP omits annotated videos
larger than 100 MiB and includes a note; the separate video download is not size
limited by this ZIP policy. Empty result sets and absent crops are represented
without fabricating data. A confirmed registration is exported only for OCR
records marked `Recognized`; uncertain OCR candidates remain separate. Vehicle
counts mean tracked-ID line-crossing events, not verified distinct real-world
vehicles. The PDF limits its detailed tracking table to the first 500 stored
observations; CSV and Excel contain the full stored observations.

New analyses preserve each tracked observation and distinguish a geometric line
crossing from whether that crossing incremented the unique tracked-ID total.
Analyses created before these fields were saved may not include class counts or
tracking records in their manifests; exports mark those records unavailable
instead of rerunning inference or inferring values from the rendered video.

---

## Running Verification Tests

### 1. Run License Plate Detection & OCR Unit Test:
```powershell
python tests/test_ocr_plate.py
```

### 2. Run ByteTrack & Line Counting Test:
```powershell
python tests/test_tracking.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Main Web Dashboard |
| `GET`  | `/api/status` | Engine status & model check |
| `POST` | `/api/process-video` | Upload video & run ByteTrack + OCR pipeline |
| `GET`  | `/api/plates` | Fetch all recognized license plate records |
| `GET`  | `/api/search?q=KL07` | Search vehicles by registration number |
