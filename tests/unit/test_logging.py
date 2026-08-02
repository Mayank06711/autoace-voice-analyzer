import json
import logging
from pathlib import Path

from app.logging_conf import bind, event, setup_logging


def test_jsonl_line_has_event_and_context(tmp_path):
    setup_logging(level="DEBUG", log_dir=str(tmp_path), as_json=True)
    log = logging.getLogger("test.mod")
    with bind(batch="b1", file="f1"):
        event(log, logging.INFO, "file.done", "ok", duration_ms=5)

    files = list(Path(tmp_path).glob("run_*.jsonl"))
    assert files, "a per-run jsonl file should be created"
    last = files[0].read_text(encoding="utf-8").strip().splitlines()[-1]
    obj = json.loads(last)
    assert obj["event"] == "file.done"
    assert obj["batch_id"] == "b1"
    assert obj["file_id"] == "f1"
    assert obj["duration_ms"] == 5


def test_event_with_reserved_key_does_not_crash(tmp_path):
    # 'name' collides with LogRecord.name — must be sanitized, not raise (regression: stuck batches)
    setup_logging(level="INFO", log_dir=str(tmp_path), as_json=True)
    log = logging.getLogger("test.reserved")
    event(log, logging.INFO, "file.started", name="call_001.ogg")  # would KeyError before the fix
    files = list(Path(tmp_path).glob("run_*.jsonl"))
    obj = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert obj["event"] == "file.started"
    assert obj.get("name_") == "call_001.ogg"
