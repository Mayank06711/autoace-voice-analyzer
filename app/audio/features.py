"""Feature extraction: build the frame-aligned timelines (energy, is_speech, spectrum) once.

All timelines share the 10 ms hop grid and are truncated to a common length T so detectors can
index them together (the alignment invariant).
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from app.config import Settings, get_settings
from app.domain.contracts import AudioClip, Features
from app.logging_conf import event

logger = logging.getLogger(__name__)

_N_FFT = 512
_RMS_FRAME = 400  # 25 ms @ 16 kHz


def _energy_dbfs(y: np.ndarray, hop: int) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=_RMS_FRAME, hop_length=hop, center=True)[0]
    return 20.0 * np.log10(np.maximum(rms, 1e-7))


def _spectral(y: np.ndarray, sr: int, hop: int) -> dict:
    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=hop, center=True))
    return {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
    }


def _vad_webrtc(y: np.ndarray, sr: int, n_frames: int, hop: int) -> np.ndarray:
    """webrtcvad on 30 ms frames → per-sample mask → sampled onto the T-frame grid."""
    import webrtcvad

    vad = webrtcvad.Vad(2)
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    frame_len = int(sr * 30 / 1000)  # 480 samples @16k
    mask = np.zeros(len(y), dtype=bool)
    for start in range(0, len(pcm16) - frame_len + 1, frame_len):
        chunk = pcm16[start:start + frame_len].tobytes()
        try:
            if vad.is_speech(chunk, sr):
                mask[start:start + frame_len] = True
        except Exception:
            pass
    if len(y) == 0:
        return np.zeros(n_frames, dtype=bool)
    centers = np.minimum(np.arange(n_frames) * hop + hop // 2, len(y) - 1)
    return mask[centers]


def _vad_energy(energy_dbfs: np.ndarray, s: Settings) -> np.ndarray:
    """Torch-free fallback VAD: speech ≈ energy well above the silence floor."""
    return energy_dbfs > (s.energy_floor_dbfs + 12.0)


_SILERO = None
_SILERO_PATH = Path("models/silero_vad.onnx")


def _get_silero():
    """Load the Silero VAD ONNX session once (torch-free, ~2.3 MB). onnxruntime is thread-safe."""
    global _SILERO
    if _SILERO is None:
        import onnxruntime as ort

        _SILERO = ort.InferenceSession(str(_SILERO_PATH), providers=["CPUExecutionProvider"])
    return _SILERO


def _vad_silero(y: np.ndarray, sr: int, n_frames: int, hop: int, s: Settings) -> np.ndarray:
    """Silero VAD (ONNX) @16 kHz, per Silero's own OnnxWrapper: prepend 64 context samples to each
    512-sample window (→576), carry the LSTM state and context. Torch-free."""
    sess = _get_silero()
    win, ctx = 512, 64
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, ctx), dtype=np.float32)
    sr_t = np.array(16000, dtype=np.int64)
    mask = np.zeros(len(y), dtype=bool)
    thr = s.vad_threshold
    for start in range(0, len(y) - win + 1, win):
        block = y[start:start + win].astype(np.float32).reshape(1, -1)
        inp = np.concatenate([context, block], axis=1)  # 64 context + 512 new = 576
        out, state = sess.run(None, {"input": inp, "state": state, "sr": sr_t})
        if float(out[0, 0]) >= thr:
            mask[start:start + win] = True
        context = inp[:, -ctx:]
    if len(y) == 0:
        return np.zeros(n_frames, dtype=bool)
    centers = np.minimum(np.arange(n_frames) * hop + hop // 2, len(y) - 1)
    return mask[centers]


def extract_features(clip: AudioClip, settings: Settings | None = None) -> Features:
    s = settings or get_settings()
    y = np.ascontiguousarray(clip.samples, dtype=np.float32)
    hop = int(s.sample_rate * s.hop_ms / 1000)

    energy = _energy_dbfs(y, hop)
    spectral = _spectral(y, clip.sr, hop)

    # common length T → alignment invariant
    t = min(len(energy), *(len(v) for v in spectral.values()))
    energy = energy[:t]
    spectral = {k: v[:t] for k, v in spectral.items()}

    # VAD backend dispatch with graceful fallback (silero ONNX → webrtc → energy).
    is_speech = None
    if s.vad_backend == "silero" and _SILERO_PATH.exists():
        try:
            is_speech = _vad_silero(y, clip.sr, t, hop, s)
        except Exception as e:
            event(logger, logging.WARNING, "vad.fallback", reason=f"silero: {e}")
    if is_speech is None and clip.sr in (8000, 16000, 32000, 48000):
        try:
            is_speech = _vad_webrtc(y, clip.sr, t, hop)
        except Exception as e:  # pragma: no cover
            event(logger, logging.WARNING, "vad.fallback", reason=str(e))
    if is_speech is None:
        is_speech = _vad_energy(energy, s)

    is_speech = np.asarray(is_speech, dtype=bool)[:t]
    if len(is_speech) < t:
        is_speech = np.pad(is_speech, (0, t - len(is_speech)))

    assert len(energy) == t and len(is_speech) == t
    assert all(len(v) == t for v in spectral.values())

    event(logger, logging.DEBUG, "features.built", frames=t)
    return Features(
        sr=clip.sr, hop_ms=s.hop_ms, energy_dbfs=energy,
        is_speech=is_speech, spectral=spectral, duration_s=clip.duration,
    )
