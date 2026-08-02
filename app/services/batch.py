"""Batch orchestration: bounded async worker pool over queued files, per-file isolation."""
from __future__ import annotations

import asyncio
import logging
import time

from app.analysis.emotion.registry import build_chain, get_provider
from app.analysis.pipeline import run_file
from app.config import Settings, get_settings
from app.errors import DecodeError
from app.infra import repositories as repo
from app.infra.storage import get_storage
from app.logging_conf import bind, event

logger = logging.getLogger(__name__)


async def process_batch(batch_id: str, s: Settings | None = None) -> None:
    """Process all queued files, ≤ APP_MAX_WORKERS at once. One failure never sinks the batch."""
    s = s or get_settings()
    store = get_storage(s)
    meta = repo.batch_status(batch_id) or {}
    # honor the per-batch provider/mode chosen in the dashboard (falls back to env defaults);
    # build_chain adds the configured secondary provider so a primary failure falls back mid-batch.
    try:
        providers = build_chain(meta.get("provider"), s, mode=meta.get("mode_emotion"))
    except Exception as e:
        event(logger, logging.ERROR, "provider.unavailable", reason=str(e))
        providers = [get_provider("mock", s)]
    sem = asyncio.Semaphore(max(1, s.max_workers))
    queued = [f for f in repo.list_files(batch_id) if f["status"] == "queued"]

    async def worker(f: dict) -> None:
        async with sem:  # gate live file-coroutines to MAX_WORKERS
            with bind(batch=batch_id, file=f["file_id"]):
                repo.set_status(f["file_id"], "processing")
                event(logger, logging.INFO, "file.started", file=f["name"])
                t = time.perf_counter()
                try:
                    with store.localize(f["storage_key"]) as path:
                        # hard per-file wall-clock cap so one stuck file never hangs the batch
                        result, cost_meta = await asyncio.wait_for(
                            run_file(path, providers, s), timeout=s.file_timeout_s + 10
                        )
                    ms = int((time.perf_counter() - t) * 1000)
                    repo.save_result(f["file_id"], result, ms=ms, meta=cost_meta)
                    event(logger, logging.INFO, "file.done", ms=ms)
                except asyncio.TimeoutError:
                    repo.mark_failed(f["file_id"], f"timeout after {s.file_timeout_s}s")
                    event(logger, logging.ERROR, "file.failed", reason="timeout")
                except DecodeError as e:
                    repo.mark_failed(f["file_id"], f"decode: {e}")
                    event(logger, logging.ERROR, "file.failed", reason=str(e))
                except Exception as e:  # isolation: keep the batch going
                    repo.mark_failed(f["file_id"], str(e))
                    event(logger, logging.ERROR, "file.failed", reason=str(e))

    await asyncio.gather(*(worker(f) for f in queued))
    event(logger, logging.INFO, "batch.finished")
