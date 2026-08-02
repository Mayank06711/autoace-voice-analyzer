"""Audio decode + validation. Robust cross-format decode via ffmpeg → 16 kHz mono float32."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.domain.contracts import AudioClip
from app.errors import DecodeError
from app.logging_conf import event

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".ogg", ".wav", ".mp3", ".flac", ".m4a", ".aac", ".opus"}


def is_supported(name: str) -> bool:
    """Cheap structural check used by Stage-1 ingestion validation."""
    return Path(name).suffix.lower() in SUPPORTED_EXTS


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise DecodeError("ffmpeg not found on PATH")
    return exe


def decode(path, settings: Settings | None = None) -> AudioClip:
    """Decode any supported file to mono float32 at the configured sample rate.

    Raises DecodeError on unsupported extension, ffmpeg failure, or empty/too-short audio
    (this is Stage-2 content validation — corruption is only knowable by trying to decode).
    """
    s = settings or get_settings()
    path = Path(path)
    if not path.exists():
        raise DecodeError(f"file not found: {path}")
    if not is_supported(path.name):
        raise DecodeError(f"unsupported extension: {path.suffix}")

    cmd = [
        _ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(s.sample_rate), "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=s.file_timeout_s)
    except subprocess.TimeoutExpired as e:
        raise DecodeError(f"decode timed out after {s.file_timeout_s}s") from e
    except Exception as e:  # pragma: no cover - environment failure
        raise DecodeError(f"ffmpeg failed to run: {e}") from e

    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore").strip()[:200]
        raise DecodeError(f"decode failed: {msg or 'ffmpeg error'}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise DecodeError("decoded to empty audio")
    duration = samples.size / float(s.sample_rate)
    if duration < 0.05:
        raise DecodeError(f"audio too short ({duration:.3f}s)")

    event(logger, logging.DEBUG, "audio.decoded", sr=s.sample_rate, duration=round(duration, 2))
    return AudioClip(samples=np.ascontiguousarray(samples), sr=s.sample_rate, duration=duration)
