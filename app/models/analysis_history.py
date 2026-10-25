from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    __table_args__ = (
        Index("ix_analysis_records_status_created_at", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    line_position_percent: Mapped[float | None] = mapped_column(Float)
    total_count: Mapped[int | None] = mapped_column(Integer)
    class_counts: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    tracking_events: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    anpr_records: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    search_results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    artifact_paths: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    failure_message: Mapped[str | None] = mapped_column(String(512))
    export_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
