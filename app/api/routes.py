"""HTTP routes: 2 HTML pages + 5 JSON endpoints. Thin — validate, call a service, shape the response."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form, HTTPException,
                     Request, Response, UploadFile)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.analysis.emotion import pricing
from app.analysis.emotion.registry import available
from app.api import auth
from app.api.deps import require_session
from app.config import get_settings
from app.errors import IngestionError
from app.infra import repositories as repo
from app.logging_conf import event
from app.services import batch, compare, ingestion, results

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


# ---------- HTML pages ----------
@router.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    # already signed in → go straight to the app (mirror of the /app guard below)
    s = get_settings()
    if auth.valid_session(request.cookies.get(auth.COOKIE), s):
        return RedirectResponse("/app")
    return templates.TemplateResponse(request, "login.html")


@router.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    s = get_settings()
    if not auth.valid_session(request.cookies.get(auth.COOKIE), s):
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "app.html")


# ---------- JSON API (auth on all) ----------
@router.post("/api/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    s = get_settings()
    if not auth.verify_login(username, password, s):
        return _err("unauthorized", "invalid username or password", 401)
    resp = JSONResponse({"ok": True})
    # secure flag on HTTPS so the session cookie isn't sent in cleartext (off for local http)
    resp.set_cookie(auth.COOKIE, auth.make_token(s), httponly=True, samesite="lax",
                    secure=(request.url.scheme == "https"))
    return resp


@router.post("/api/batches")
async def create_batch(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    mode: str = Form(None),
    provider: str = Form(None),
    _=Depends(require_session),
):
    """Accept EITHER a single .zip OR many files (a folder upload, per brief §7)."""
    import os
    import shutil

    s = get_settings()
    if provider and provider.lower() not in available():
        return _err("bad_provider", f"unknown provider {provider!r}; have {available()}", 400)
    cap = s.max_upload_mb * 1024 * 1024
    emode = mode or s.emotion_mode
    eprov = provider or s.emotion_provider

    try:
        if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
            data = await files[0].read()
            if len(data) > cap:
                return _err("too_large", f"upload exceeds {s.max_upload_mb} MB", 413)
            fd, tmpname = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            tmp = Path(tmpname)
            tmp.write_bytes(data)
            try:
                rep = ingestion.ingest_zip(tmp, emode, eprov, s)
            finally:
                tmp.unlink(missing_ok=True)
        else:  # folder upload — save the files to a temp dir, then ingest the dir
            tmpdir = Path(tempfile.mkdtemp(prefix="upload_"))
            try:
                total = 0
                for uf in files:
                    chunk = await uf.read()
                    total += len(chunk)
                    if total > cap:
                        return _err("too_large", f"upload exceeds {s.max_upload_mb} MB", 413)
                    name = Path(uf.filename or "").name  # basename (ignore browser relative paths)
                    if name:
                        (tmpdir / name).write_bytes(chunk)
                rep = ingestion.ingest_dir(tmpdir, emode, eprov, s)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
    except IngestionError as e:
        return _err("ingestion", str(e), 400)

    background.add_task(batch.process_batch, rep["batch_id"])
    return JSONResponse(rep, status_code=202)


@router.get("/api/batches")
def list_batches(_=Depends(require_session)):
    return {"batches": repo.list_batches()}


@router.get("/api/batches/{batch_id}/status")
def status(batch_id: str, _=Depends(require_session)):
    st = repo.batch_status(batch_id)
    if not st:
        raise HTTPException(status_code=404, detail="batch not found")
    # attach the per-1M rate card for the model(s) that ran → dashboard shows the cost BASIS
    st["rates"] = {m: pricing.rates_dict(m) for m in st.get("models", [])}
    return st


@router.get("/api/batches/{batch_id}/results")
def get_results(batch_id: str, _=Depends(require_session)):
    if not repo.batch_exists(batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    files = repo.list_files(batch_id)
    return {
        "batch_id": batch_id,
        "results": [
            {"name": f["name"],
             "result_json": json.loads(f["result_json"]) if f["result_json"] else None,
             "error": f["error"],
             # cost telemetry (kept separate from the 9-field result_json)
             "model": f["emotion_model"], "provider": f["emotion_provider"],
             "audio_tokens": f["audio_tokens"], "text_tokens": f["text_tokens"],
             "cost_usd": f["cost_usd"], "audio_seconds": f["audio_seconds"]}
            for f in files
        ],
    }


@router.post("/api/batches/{batch_id}/compare")
def compare_labels(batch_id: str, _=Depends(require_session)):
    """Predictions vs labels.csv ground truth: exact + LLM-semantic agreement (labeled files only).

    Sync def → FastAPI runs it in a threadpool, so the blocking compare LLM call never stalls the loop.
    """
    if not repo.batch_exists(batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    return compare.run_comparison(batch_id)


@router.get("/api/batches/{batch_id}/download")
def download(batch_id: str, fmt: str = "json", _=Depends(require_session)):
    if not repo.batch_exists(batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    media, name, body = results.export(batch_id, "csv" if fmt == "csv" else "json")
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
