"""long_silence_present — longest run of non-speech + low-energy frames vs a threshold."""
from __future__ import annotations

from app.domain.contracts import Features


def detect(features: Features, audio, s) -> dict:
    quiet = (~features.is_speech) & (features.energy_dbfs < s.energy_floor_dbfs)
    longest = cur = 0
    for q in quiet:
        cur = cur + 1 if q else 0
        if cur > longest:
            longest = cur
    dur_s = longest * features.hop_ms / 1000.0
    return {"long_silence_present": bool(dur_s >= s.long_silence_s)}
