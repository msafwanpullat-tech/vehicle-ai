import os
import sys
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is in Python path for clean module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.routes.counting_routes import router as counting_router
from app.routes.video_routes import router as video_router
from app.database import verify_database_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    verify_database_connection()
    yield


# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(
    title="AI-Based Vehicle Detection, Classification and Counting System",
    description="YOLO11 vehicle detection, tracking, and line-crossing counts",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for browser client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
for folder in ["uploads", "outputs", "static", "models", "videos"]:
    (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)

# Mount static files directories
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=PROJECT_ROOT / "uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory=PROJECT_ROOT / "outputs"), name="outputs")

# Setup HTML template renderer
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")

# Register API routes
app.include_router(video_router)
app.include_router(counting_router)


@app.get("/")
def read_root(request: Request):
    """Render the vehicle detection and counting dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="counting_dashboard.html",
        context={"title": app.title},
    )
