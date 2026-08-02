"""Repository layer — the only place that touches the DB. Services/API never write SQL."""
from __future__ import annotations

import uuid

from sqlalchemy import func, select

from app.domain.schema import AnalysisResult
from app.infra.db import get_session
from app.infra.models import Batch, File


def new_id() -> str:
    return uuid.uuid4().hex


def create_batch(mode: str, provider: str, files: list[tuple]) -> str:
    """files = list of (name, storage_key) or (name, storage_key, label_json). Returns batch_id."""
    bid = new_id()
    with get_session() as db:
        db.add(Batch(batch_id=bid, status="processing", mode_emotion=mode,
                     provider=provider, total_files=len(files)))
        for f in files:
            name, key = f[0], f[1]
            label = (f[2] if len(f) > 2 else None) or None
            db.add(File(file_id=new_id(), batch_id=bid, original_name=name,
                        storage_key=key, status="queued", label_json=label))
        db.commit()
    return bid


def comparison_pairs(batch_id: str) -> list[dict]:
    """Files that have BOTH a prediction and a ground-truth label — for the labels-vs-us comparison."""
    with get_session() as db:
        rows = db.scalars(select(File).where(File.batch_id == batch_id)).all()
        out = []
        for f in rows:
            if f.result_json and f.label_json:
                out.append({"name": f.original_name, "predicted": f.result_json, "label": f.label_json})
        return out


def list_files(batch_id: str) -> list[dict]:
    with get_session() as db:
        rows = db.scalars(select(File).where(File.batch_id == batch_id)).all()
        return [
            {"file_id": f.file_id, "name": f.original_name, "storage_key": f.storage_key,
             "status": f.status, "error": f.error_reason, "result_json": f.result_json,
             "emotion_provider": f.emotion_provider, "emotion_model": f.emotion_model,
             "audio_tokens": f.audio_tokens, "text_tokens": f.text_tokens,
             "cost_usd": f.cost_usd, "audio_seconds": f.audio_seconds}
            for f in rows
        ]


def set_status(file_id: str, status: str, error: str | None = None) -> None:
    with get_session() as db:
        f = db.get(File, file_id)
        if f:
            f.status = status
            if error:
                f.error_reason = error
            db.commit()


def save_result(file_id: str, result: AnalysisResult, ms: int | None = None,
                meta: dict | None = None) -> None:
    """Idempotent per file_id: upserting a result twice leaves one row.

    `meta` (optional) is the emotion cost telemetry from the pipeline: provider/model/tokens/$.
    """
    with get_session() as db:
        f = db.get(File, file_id)
        if not f:
            return
        f.result_json = result.model_dump_json()
        f.status = "done"
        f.processing_ms = ms
        f.error_reason = None
        if meta:
            f.emotion_provider = meta.get("provider") or None
            f.emotion_model = meta.get("model") or None
            f.audio_tokens = meta.get("audio_tokens")
            f.text_tokens = (meta.get("text_in_tokens") or 0) + (meta.get("text_out_tokens") or 0)
            f.cost_usd = meta.get("cost_usd")
            f.audio_seconds = meta.get("audio_seconds")
        _recount(db, f.batch_id)
        db.commit()


def mark_failed(file_id: str, reason: str) -> None:
    with get_session() as db:
        f = db.get(File, file_id)
        if not f:
            return
        f.status = "failed"
        f.error_reason = reason
        _recount(db, f.batch_id)
        db.commit()


def _recount(db, batch_id: str) -> None:
    b = db.get(Batch, batch_id)
    if not b:
        return
    files = db.scalars(select(File).where(File.batch_id == batch_id)).all()
    b.done_count = sum(1 for f in files if f.status == "done")
    b.failed_count = sum(1 for f in files if f.status == "failed")
    if b.done_count + b.failed_count >= b.total_files:
        b.status = "done"


def batch_status(batch_id: str) -> dict | None:
    with get_session() as db:
        b = db.get(Batch, batch_id)
        if not b:
            return None
        files = db.scalars(select(File).where(File.batch_id == batch_id)).all()
        cost = _cost_summary(files)
        return {
            "batch_id": b.batch_id, "status": b.status, "total": b.total_files,
            "done": b.done_count, "failed": b.failed_count,
            "provider": b.provider, "mode_emotion": b.mode_emotion,
            "has_labels": any(f.label_json for f in files),  # enables the vs-labels comparison
            **cost,  # cost_usd, audio_seconds, cost_per_min, models
            "files": [{"name": f.original_name, "status": f.status, "error": f.error_reason}
                      for f in files],
        }


def _cost_summary(files) -> dict:
    """Aggregate real emotion spend across a batch's files (for the dashboard cost readout).

    `models`/`providers` are what ACTUALLY ran (post-fallback), so the UI can flag when the served
    provider differs from the one requested.
    """
    total_cost = round(sum(f.cost_usd or 0.0 for f in files), 6)
    total_sec = round(sum(f.audio_seconds or 0.0 for f in files), 2)
    models = sorted({f.emotion_model for f in files if f.emotion_model})
    providers = sorted({f.emotion_provider for f in files if f.emotion_provider})
    return {
        "cost_usd": total_cost,
        "audio_seconds": total_sec,
        "cost_per_min": round(total_cost / (total_sec / 60.0), 6) if total_sec > 0 else 0.0,
        "models": models,
        "providers": providers,
    }


def list_batches(limit: int = 50) -> list[dict]:
    """Recent batches (newest first) for the dashboard history."""
    with get_session() as db:
        rows = db.scalars(select(Batch).order_by(Batch.created_at.desc()).limit(limit)).all()
        cost_by = dict(db.execute(  # total emotion spend per batch, one grouped query
            select(File.batch_id, func.coalesce(func.sum(File.cost_usd), 0.0)).group_by(File.batch_id)
        ).all())
        model_by = dict(db.execute(  # actual model(s) that ran per batch (post-fallback)
            select(File.batch_id, func.group_concat(File.emotion_model.distinct()))
            .where(File.emotion_model.isnot(None)).group_by(File.batch_id)
        ).all())
        return [
            {"batch_id": b.batch_id, "created_at": b.created_at.isoformat() + "Z",
             "status": b.status, "total": b.total_files, "done": b.done_count,
             "failed": b.failed_count, "provider": b.provider,
             "cost_usd": round(cost_by.get(b.batch_id, 0.0), 6),
             "models": [m for m in (model_by.get(b.batch_id) or "").split(",") if m]}
            for b in rows
        ]


def batch_exists(batch_id: str) -> bool:
    with get_session() as db:
        return db.get(Batch, batch_id) is not None


def batch_ids_with_queued() -> list[str]:
    """Batch ids that still have queued files (used to resume after a restart)."""
    with get_session() as db:
        rows = db.scalars(select(File.batch_id).where(File.status == "queued")).all()
        return sorted(set(rows))


def sweep_processing() -> int:
    """On startup, requeue any files stuck in 'processing' from a previous crash."""
    with get_session() as db:
        rows = db.scalars(select(File).where(File.status == "processing")).all()
        for f in rows:
            f.status = "queued"
        db.commit()
        return len(rows)
