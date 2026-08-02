"""SQLite engine/session. WAL mode + one shared factory; safe for the single-process worker pool."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy import event as sa_event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_Session = None


def init_db(url: str | None = None):
    """Create the engine (WAL), tables, and session factory. Idempotent-ish; url override for tests."""
    global _engine, _Session
    s = get_settings()
    db_url = url or s.db_url
    if db_url.startswith("sqlite:///"):  # ensure the sqlite parent dir exists (fresh container)
        parent = os.path.dirname(db_url[len("sqlite:///"):])
        if parent:
            os.makedirs(parent, exist_ok=True)
    _engine = create_engine(db_url, connect_args={"check_same_thread": False}, future=True)

    @sa_event.listens_for(_engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _rec):  # WAL = concurrent reads while a write is in flight
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    from app.infra import models  # noqa: F401  register mapped classes

    Base.metadata.create_all(_engine)
    _ensure_columns(_engine)  # add newly-introduced columns to a pre-existing DB (create_all won't)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


# Columns added after the first release → ALTER them onto older SQLite files (idempotent).
_ADDED_FILE_COLUMNS = {
    "emotion_provider": "VARCHAR",
    "emotion_model": "VARCHAR",
    "audio_tokens": "INTEGER",
    "text_tokens": "INTEGER",
    "cost_usd": "FLOAT",
    "audio_seconds": "FLOAT",
    "label_json": "TEXT",
}


def _ensure_columns(engine) -> None:
    """Lightweight forward-only migration for SQLite: add any missing `files` columns."""
    from sqlalchemy import text
    with engine.begin() as conn:
        have = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(files)")}
        for col, sqltype in _ADDED_FILE_COLUMNS.items():
            if col not in have:
                conn.exec_driver_sql(f"ALTER TABLE files ADD COLUMN {col} {sqltype}")


def get_session():
    if _Session is None:
        init_db()
    return _Session()
