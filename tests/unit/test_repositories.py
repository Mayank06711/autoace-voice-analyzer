from app.config import get_settings
from app.domain.schema import AnalysisResult
from app.infra import repositories as repo
from app.infra.db import init_db
from app.infra.storage import LocalStorage

RESULT = AnalysisResult.repair(
    {"emotional_tone": "neutral", "emotional_intensity": "low", "background_noise_present": False,
     "background_noise_severity": "none", "audio_quality": "clear", "speaker_overlap_present": False,
     "long_silence_present": False, "confidence": 0.5}
)


def _init(tmp_path):
    init_db(f"sqlite:///{tmp_path.as_posix()}/t.db")


def test_batch_crud_counts_and_idempotent_result(tmp_path):
    _init(tmp_path)
    bid = repo.create_batch("api", "mock", [("a.ogg", "k/a.ogg"), ("b.ogg", "k/b.ogg")])
    files = repo.list_files(bid)
    assert len(files) == 2

    repo.save_result(files[0]["file_id"], RESULT, ms=10)
    repo.save_result(files[0]["file_id"], RESULT, ms=10)  # idempotent → still one row
    assert len(repo.list_files(bid)) == 2
    assert repo.batch_status(bid)["done"] == 1

    repo.mark_failed(files[1]["file_id"], "corrupt")
    st = repo.batch_status(bid)
    assert st["failed"] == 1
    assert st["status"] == "done"  # done + failed == total


def test_sweep_requeues_processing(tmp_path):
    _init(tmp_path)
    bid = repo.create_batch("api", "mock", [("a.ogg", "k/a.ogg")])
    fid = repo.list_files(bid)[0]["file_id"]
    repo.set_status(fid, "processing")
    assert repo.sweep_processing() == 1
    assert repo.list_files(bid)[0]["status"] == "queued"


def test_local_storage_roundtrip(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "storage_dir", str(tmp_path))
    store = LocalStorage(s)
    key = store.save("k/a.bin", b"hello")
    assert store.exists(key)
    assert store.open(key) == b"hello"
    with store.localize(key) as p:
        with open(p, "rb") as fh:
            assert fh.read() == b"hello"
