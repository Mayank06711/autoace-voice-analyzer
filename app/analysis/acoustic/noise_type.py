"""background_noise_type — coarse spectral heuristic (YAMNet-ONNX is the documented upgrade).

Merge clears this to "" when background_noise_present is False, so a guess here is harmless.
"""
from __future__ import annotations

import numpy as np

from app.domain.contracts import Features


def detect(features: Features, audio, s) -> dict:
    if not s.noise_type_enabled:
        return {"background_noise_type": ""}

    non_speech = ~features.is_speech
    if int(non_speech.sum()) < 3:
        return {"background_noise_type": ""}

    flatness = float(np.median(features.spectral["flatness"][non_speech]))
    centroid = float(np.median(features.spectral["centroid"][non_speech]))
    rolloff = float(np.median(features.spectral["rolloff"][non_speech]))

    # Coarse source guess from spectral SHAPE (all we can infer without an event classifier):
    #   flat + broadband  -> static/hiss   |   energy low & tonal -> hum/mechanical
    #   energy mostly high -> high-freq hiss |   mid-band, structured -> indistinct ambient sound
    # A spectral heuristic CANNOT reliably name a source (e.g. tell a TV from static — both sit in the
    # mid band); YAMNet-ONNX is the documented upgrade for true source labels. So the mid-band case is
    # reported honestly as generic "ambient noise", not an over-specific guess.
    if flatness > 0.20:
        label = "static/hiss"
    elif centroid < 400:
        label = "hum/mechanical"
    elif centroid > 2800 and rolloff > 5000:
        label = "high-frequency hiss"
    else:
        label = "ambient noise"
    return {"background_noise_type": label}
