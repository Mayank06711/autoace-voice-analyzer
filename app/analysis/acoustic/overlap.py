"""speaker_overlap_present — conservative torch-free heuristic.

Documented as approximate: real accuracy comes from the `llm` overlap method (provider returns it)
or the pyannote upgrade. In `heuristic` mode we flag overlap only on sustained spectral crowding
within speech, which correlates with two simultaneous voices.
"""
from __future__ import annotations

import numpy as np

from app.domain.contracts import Features


def detect(features: Features, audio, s) -> dict:
    speech = features.is_speech
    if int(speech.sum()) < 5:
        return {"speaker_overlap_present": False}
    flat_speech = features.spectral["flatness"][speech]
    frac_crowded = float(np.mean(flat_speech > 0.10))
    return {"speaker_overlap_present": bool(frac_crowded > 0.40)}
