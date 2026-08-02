"""ORM: Batch (1) ──< File (many). Small rows only; audio blobs live in StorageBackend."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infra.db import Base


class Batch(Base):
    __tablename__ = "batches"

    batch_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String, default="validating")  # validating|processing|done|failed
    mode_emotion: Mapped[str] = mapped_column(String, default="api")
    provider: Mapped[str] = mapped_column(String, default="mock")
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    done_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)

    files: Mapped[list["File"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class File(Base):
    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String, primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batches.batch_id"), index=True)
    original_name: Mapped[str] = mapped_column(String)
    storage_key: Mapped[str] = mapped_column(String)  # StorageBackend key, not an assumed disk path
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|processing|done|failed
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    label_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # ground truth from labels.csv
    error_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- emotion cost telemetry (metadata, NOT part of the 9-field result_json) ---
    emotion_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    emotion_model: Mapped[str | None] = mapped_column(String, nullable=True)
    audio_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    batch: Mapped["Batch"] = relationship(back_populates="files")
