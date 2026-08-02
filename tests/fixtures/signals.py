"""Synthetic-signal generators for deterministic unit tests (no real audio needed)."""
from __future__ import annotations

import numpy as np
import soundfile as sf


def write_wav(path, y: np.ndarray, sr: int = 16000) -> str:
    sf.write(str(path), y.astype(np.float32), sr, subtype="FLOAT")
    return str(path)


def silence(dur: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(dur * sr), dtype=np.float32)


def tone(freq: float = 220.0, dur: float = 1.0, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def white_noise(dur: float = 1.0, sr: int = 16000, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(dur * sr))).astype(np.float32)


def clipped_tone(freq: float = 220.0, dur: float = 1.0, sr: int = 16000) -> np.ndarray:
    """A loud tone hard-clipped to [-1, 1] — exercises the clipping detector."""
    return np.clip(tone(freq, dur, sr, amp=2.0), -1.0, 1.0).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)
