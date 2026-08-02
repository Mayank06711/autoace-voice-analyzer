"""Stage-1 structural validation + blob persistence → a processing-ready batch.

Accepts a folder or a ZIP; finds audio (root or one nested subfolder); parses the optional
labels.csv manifest; matches names; saves blobs to the StorageBackend; creates Batch/File rows.
"""
from __future__ import annotations

import csv
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from app.audio.io import is_supported
from app.config import Settings, get_settings
from app.errors import IngestionError
from app.infra import repositories as repo
from app.infra.storage import get_storage
from app.logging_conf import event

logger = logging.getLogger(__name__)


def _find_audio_dir(root: Path) -> Path:
    if any(p.is_file() and is_supported(p.name) for p in root.iterdir()):
        return root
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else root  # descend a single wrapper folder


def _parse_manifest(d: Path) -> dict | None:
    m = d / "labels.csv"
    if not m.exists():
        return None
    with open(m, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise IngestionError("labels.csv must contain a 'name' column")
        return {r["name"]: (r.get("result_json") or "") for r in reader if r.get("name")}


def ingest_dir(root: Path, mode: str, provider: str, s: Settings | None = None) -> dict:
    s = s or get_settings()
    d = _find_audio_dir(root)
    audio = sorted(p for p in d.iterdir() if p.is_file() and is_supported(p.name))
    if not audio:
        raise IngestionError("no supported audio files found in upload")

    manifest = _parse_manifest(d)
    names = {p.name for p in audio}
    unmatched = sorted(n for n in names if manifest is not None and n not in manifest)
    missing = sorted(n for n in (manifest or {}) if n not in names)

    store = get_storage(s)
    prefix = repo.new_id()
    files: list[tuple] = []
    for p in audio:
        key = f"{prefix}/{p.name}"
        store.save(key, p.read_bytes())
        label = manifest.get(p.name) if manifest else None  # ground-truth result_json, if provided
        files.append((p.name, key, label))

    batch_id = repo.create_batch(mode, provider, files)
    event(logger, logging.INFO, "batch.created", total=len(files), has_manifest=manifest is not None)
    return {
        "batch_id": batch_id,
        "total": len(files),
        "validation": {
            "matched": sorted(n for n in names if manifest is None or n in manifest),
            "unmatched_files": unmatched,
            "missing_audio": missing,
            "has_manifest": manifest is not None,
        },
    }


def ingest_zip(zip_path: Path, mode: str, provider: str, s: Settings | None = None) -> dict:
    s = s or get_settings()
    tmp = Path(tempfile.mkdtemp(prefix="batch_"))
    max_uncompressed = s.max_upload_mb * 1024 * 1024 * 4  # guard against zip bombs
    try:
        with zipfile.ZipFile(zip_path) as z:
            infos = z.infolist()
            if len(infos) > 5000:
                raise IngestionError("archive has too many entries")
            if sum(i.file_size for i in infos) > max_uncompressed:
                raise IngestionError("archive uncompressed size too large")
            for i in infos:  # zip-slip: reject absolute or parent-escaping paths
                parts = Path(i.filename).parts
                if i.filename.startswith(("/", "\\")) or ".." in parts:
                    raise IngestionError(f"unsafe path in archive: {i.filename}")
            z.extractall(tmp)
    except zipfile.BadZipFile as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise IngestionError(f"not a valid zip archive: {e}") from e
    try:
        return ingest_dir(tmp, mode, provider, s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
