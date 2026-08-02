"""background_noise_present + background_noise_severity via WADA-SNR on speech-active audio.

WADA detects noise sitting UNDER speech (no silent gap needed). We gate on 'has speech' so a silent
clip is never flagged as noisy. Thresholds are calibrated on the 3 labeled calls; the hidden set is
the real evaluation (see docs/VALIDATION.md).
"""
from __future__ import annotations

from app.analysis.acoustic._dsp import speech_samples, wada_snr
from app.domain.contracts import Features

_ABSENT = {"background_noise_present": False, "background_noise_severity": "none"}


def detect(features: Features, audio, s) -> dict:
    if int(features.is_speech.sum()) < 5:
        return dict(_ABSENT)  # no speech to be contaminated → not "background noise"

    snr = wada_snr(speech_samples(features, audio))
    if snr >= s.noise_wada_present_db:
        return dict(_ABSENT)

    if snr >= s.noise_wada_low_db:
        severity = "low"
    elif snr >= s.noise_wada_med_db:
        severity = "medium"
    else:
        severity = "high"
    return {"background_noise_present": True, "background_noise_severity": severity}
