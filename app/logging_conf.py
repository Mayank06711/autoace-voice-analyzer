"""Structured logging: console (human) + per-run JSONL, with run/batch/file context."""
from __future__ import annotations

import contextvars
import datetime
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

_run = contextvars.ContextVar("run_id", default=None)
_batch = contextvars.ContextVar("batch_id", default=None)
_file = contextvars.ContextVar("file_id", default=None)
_VARS = {"run": _run, "batch": _batch, "file": _file}

# standard LogRecord attributes to skip when serialising extras
_STD = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"run_id", "batch_id", "file_id", "event", "message", "asctime", "taskName"}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run.get()
        record.batch_id = _batch.get()
        record.file_id = _file.get()
        if not hasattr(record, "event"):
            record.event = ""
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", ""),
            "run_id": getattr(record, "run_id", None),
            "batch_id": getattr(record, "batch_id", None),
            "file_id": getattr(record, "file_id", None),
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD and k not in base:
                base[k] = v
        if record.exc_info:
            base["error"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = ""
        if getattr(record, "file_id", None):
            ctx = f" [file={record.file_id}]"
        elif getattr(record, "batch_id", None):
            ctx = f" [batch={record.batch_id}]"
        ev = getattr(record, "event", "") or ""
        return f"{record.levelname:<7} {record.name} {ev} {record.getMessage()}{ctx}".rstrip()


def setup_logging(level: str = "INFO", log_dir: str = "logs", as_json: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()  # idempotent: safe to call again (tests, reload)
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(HumanFormatter())
    ch.addFilter(ContextFilter())
    root.addHandler(ch)

    if as_json:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"run_{ts}.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh.addFilter(ContextFilter())
        root.addHandler(fh)


@contextmanager
def bind(**ctx):
    """Scope run/batch/file ids onto every log line emitted inside the block."""
    tokens = {}
    for k, v in ctx.items():
        if k in _VARS:
            tokens[_VARS[k]] = _VARS[k].set(v)
    try:
        yield
    finally:
        for var, tok in tokens.items():
            var.reset(tok)


# LogRecord reserved attribute names — passing any of these via `extra` raises KeyError.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def event(logger: logging.Logger, level: int, event_name: str, msg: str = "", **kv) -> None:
    """The structured-log helper used everywhere: event(log, INFO, 'file.done', ms=5).

    Keys that collide with reserved LogRecord attributes (e.g. name) are suffixed with '_' so a
    stray field name can never crash logging.
    """
    safe = {(k + "_" if k in _RESERVED else k): v for k, v in kv.items()}
    logger.log(level, msg, extra={"event": event_name, **safe})
