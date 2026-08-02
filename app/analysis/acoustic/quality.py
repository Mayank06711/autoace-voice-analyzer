"""audio_quality — technical fidelity ONLY (clipping, muffling, volume).

Deliberately independent of background noise (the brief scores them separately): a clean voice with
a loud TV behind it is still audio_quality=clear. So SNR is NOT used here.
"""
from __future__ import annotations

import numpy as np

from app.domain.contracts import Features


def detect(features: Features, audio: np.ndarray, s) -> dict:
    issues = 0
    severe = False

    # distortion / clipping
    clip_ratio = float(np.mean(np.abs(audio) >= 0.99)) if audio.size else 0.0
    if clip_ratio >= s.clip_bad:
        severe = True
    elif clip_ratio >= s.clip_warn:
        issues += 1

    speech = features.is_speech
    if int(speech.sum()) > 3:
        # muffling: energy band-limited to very low frequencies
        rolloff = float(np.median(features.spectral["rolloff"][speech]))
        if rolloff < s.muffle_bad_hz:
            severe = True
        elif rolloff < s.muffle_warn_hz:
            issues += 1
        # extremely low volume
        if float(np.median(features.energy_dbfs[speech])) < s.low_volume_dbfs:
            issues += 1

    if severe:
        quality = "severely_impaired"
    elif issues >= 1:
        quality = "slightly_impaired"
    else:
        quality = "clear"
    return {"audio_quality": quality}
