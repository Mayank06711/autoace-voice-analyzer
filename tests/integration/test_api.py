import io
import time
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import get_settings
from app.infra.db import init_db


def _client(tmp_path, monkeypatch):
    init_db(f"sqlite:///{tmp_path.as_posix()}/t.db")
    s = get_settings()
    monkeypatch.setattr(s, "storage_dir", str(tmp_path / "store"))
    monkeypatch.setattr(s, "admin_key", "secret")
    monkeypatch.setattr(s, "emotion_provider", "mock")
    from app.main import app
    return TestClient(app)


def _zip_one():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(Path("data") / "call_001.ogg", arcname="call_001.ogg")
    buf.seek(0)
    return buf


def test_auth_required_on_every_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/batches/x/status").status_code == 401
    assert c.get("/api/batches/x/results").status_code == 401
    assert c.get("/api/batches/x/download").status_code == 401  # download guarded too
    assert c.post("/api/login", data={"username": "admin", "password": "wrong"}).status_code == 401


def test_login_upload_status_download(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/login", data={"username": "admin", "password": "secret"}).status_code == 200

    r = c.post("/api/batches",
               files=[("files", ("d.zip", _zip_one(), "application/zip"))],
               data={"provider": "mock"})
    assert r.status_code == 202
    bid = r.json()["batch_id"]
    assert r.json()["total"] == 1

    st = {}
    for _ in range(50):
        st = c.get(f"/api/batches/{bid}/status").json()
        if st["status"] == "done":
            break
        time.sleep(0.1)
    assert st["status"] == "done"

    res = c.get(f"/api/batches/{bid}/results").json()
    assert len(res["results"]) == 1
    assert res["results"][0]["result_json"] is not None

    dl = c.get(f"/api/batches/{bid}/download?fmt=csv")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("text/csv")


def test_folder_upload_multiple_files(tmp_path, monkeypatch):
    # brief §7: a FOLDER upload = many files (not a zip) -> ingest_dir path
    c = _client(tmp_path, monkeypatch)
    c.post("/api/login", data={"username": "admin", "password": "secret"})
    audio = (Path("data") / "call_001.ogg").read_bytes()
    r = c.post("/api/batches",
               files=[("files", ("call_001.ogg", audio, "audio/ogg")),
                      ("files", ("labels.csv", b"name,result_json\ncall_001.ogg,\n", "text/csv"))],
               data={"provider": "mock"})
    assert r.status_code == 202
    assert r.json()["total"] == 1
    assert r.json()["validation"]["has_manifest"] is True


def test_batch_history_listing(tmp_path, monkeypatch):
    # GET /api/batches powers the "Recent batches" dashboard panel — persists across page reloads.
    c = _client(tmp_path, monkeypatch)
    c.post("/api/login", data={"username": "admin", "password": "secret"})
    assert c.get("/api/batches").json()["batches"] == []  # empty before any upload

    r = c.post("/api/batches",
               files=[("files", ("d.zip", _zip_one(), "application/zip"))],
               data={"provider": "mock"})
    bid = r.json()["batch_id"]

    batches = c.get("/api/batches").json()["batches"]
    assert len(batches) == 1
    b = batches[0]
    assert b["batch_id"] == bid
    assert b["provider"] == "mock"
    assert b["created_at"].endswith("Z")  # ISO UTC so the browser can render a local timestamp
    assert {"status", "total", "done", "failed"} <= b.keys()


def test_unknown_batch_and_bad_provider(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/api/login", data={"username": "admin", "password": "secret"})
    # unknown batch -> 404 (not empty 200) on results and download
    assert c.get("/api/batches/nope/results").status_code == 404
    assert c.get("/api/batches/nope/download").status_code == 404
    # unknown provider rejected at upload
    r = c.post("/api/batches",
               files=[("files", ("d.zip", _zip_one(), "application/zip"))],
               data={"provider": "not-a-provider"})
    assert r.status_code == 400
