"""FastAPI app: serves UI + API on one port. ONE uvicorn worker (see CLAUDE.md invariants)."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.infra import repositories as repo
from app.infra.db import init_db
from app.logging_conf import event, setup_logging
from app.services import batch

_DEFAULT_ADMIN = "changeme"
_DEFAULT_SECRET = "dev-secret-change-me"

logger = logging.getLogger(__name__)


def warmup() -> None:
    """Load/JIT-warm models ONCE at startup so the first request is warm."""
    import numpy as np

    from app.audio.features import extract_features
    from app.domain.contracts import AudioClip

    try:
        extract_features(AudioClip(np.zeros(16000, np.float32), 16000, 1.0), get_settings())
        event(logger, logging.INFO, "model.loaded", component="features+vad")
    except Exception as e:  # pragma: no cover
        event(logger, logging.WARNING, "warmup.failed", reason=str(e))


def _check_secrets(s) -> None:
    if s.admin_key == _DEFAULT_ADMIN or s.session_secret == _DEFAULT_SECRET:
        event(logger, logging.CRITICAL, "insecure.defaults",
              msg="default admin_key/session_secret in use — set APP_ADMIN_KEY and APP_SESSION_SECRET")
        if s.env == "production":
            raise RuntimeError("refusing to start with default credentials in production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, s.log_json)
    _check_secrets(s)
    init_db()
    requeued = repo.sweep_processing()
    if requeued:
        event(logger, logging.INFO, "startup.sweep", requeued=requeued)
        for bid in repo.batch_ids_with_queued():  # resume any batch left unfinished by a crash
            asyncio.create_task(batch.process_batch(bid))
    warmup()
    event(logger, logging.INFO, "startup.ready", port=s.port)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AutoAce Voice Tone & Background Noise Analyzer", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.include_router(router)
    return app


app = create_app()
