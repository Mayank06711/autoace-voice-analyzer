import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.domain.schema import AnalysisResult
from app.infra import repositories as repo
from app.infra.db import init_db
from app.services import batch, ingestion, results

VALID = {
    "emotional_tone": "neutral", "emotional_intensity": "low", "background_noise_present": False,
    "background_noise_severity": "none", "audio_quality": "clear", "speaker_overlap_present": False,
    "long_silence_present": False, "confidence": 0.5,
}


def _setup(tmp_path, monkeypatch):
    init_db(f"sqlite:///{tmp_path.as_posix()}/t.db")
    s = get_settings()
    monkeypatch.setattr(s, "storage_dir", str(tmp_path / "store"))
    return s


def test_ingest_and_process_real_data(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    rep = ingestion.ingest_dir(Path("data"), "api", "mock", s)  # data/ has 3 oggs + labels.csv
    assert rep["total"] == 3
    assert rep["validation"]["has_manifest"] is True

    asyncio.run(batch.process_batch(rep["batch_id"], s))
    st = repo.batch_status(rep["batch_id"])
    assert st["done"] == 3
    assert st["status"] == "done"

    media, name, body = results.export(rep["batch_id"], "csv")
    assert media == "text/csv" and body.count(b"\n") >= 3


def test_corrupt_file_isolated(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    d = tmp_path / "in"
    d.mkdir()
    (d / "good.wav").write_bytes((Path("data") / "call_001.ogg").read_bytes())  # decodable
    (d / "bad.wav").write_bytes(b"not audio")
    # give bad.wav a real audio ext so it's 'supported' but undecodable
    rep = ingestion.ingest_dir(d, "api", "mock", s)
    asyncio.run(batch.process_batch(rep["batch_id"], s))
    st = repo.batch_status(rep["batch_id"])
    assert st["done"] == 1
    assert st["failed"] == 1
    assert st["status"] == "done"


def test_concurrency_cap_never_exceeds_max_workers(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(s, "max_workers", 3)
    files = [(f"f{i}.ogg", f"k/f{i}.ogg") for i in range(20)]
    bid = repo.create_batch("api", "mock", files)

    state = {"cur": 0, "max": 0}

    async def fake_run_file(path, providers, ss):
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        await asyncio.sleep(0.02)
        state["cur"] -= 1
        return AnalysisResult.repair(VALID), {"model": "mock", "cost_usd": 0.0}

    monkeypatch.setattr(batch, "run_file", fake_run_file)
    asyncio.run(batch.process_batch(bid, s))
    assert state["max"] <= 3
    assert repo.batch_status(bid)["done"] == 20
