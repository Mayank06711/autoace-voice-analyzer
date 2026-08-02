"""The output contract: the exact 9-field schema + enums + a repair() that never crashes."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator


class EmotionalTone(str, Enum):
    neutral = "neutral"
    satisfied = "satisfied"
    frustrated = "frustrated"
    upset = "upset"
    distressed = "distressed"


class Intensity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class NoiseSeverity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


class AudioQuality(str, Enum):
    clear = "clear"
    slightly_impaired = "slightly_impaired"
    severely_impaired = "severely_impaired"


def _coerce(value, enum, default: Enum) -> str:
    try:
        return enum(value).value
    except Exception:
        return default.value


class AnalysisResult(BaseModel):
    """One clip's result. Field names/enums match the brief exactly."""

    emotional_tone: EmotionalTone
    emotional_intensity: Intensity
    background_noise_present: bool
    background_noise_type: str = ""
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @classmethod
    def repair(cls, data: dict | None) -> "AnalysisResult":
        """Coerce any dict (e.g. a malformed provider response) into a valid result.

        Unknown enum values fall back to a safe default; confidence is clamped; and the
        cross-field rule (no noise -> empty type + severity 'none') is enforced.
        """
        d = dict(data or {})
        present = bool(d.get("background_noise_present", False))
        try:
            conf = float(d.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5  # non-numeric confidence -> safe default (never crash)
        out = {
            "emotional_tone": _coerce(d.get("emotional_tone"), EmotionalTone, EmotionalTone.neutral),
            "emotional_intensity": _coerce(d.get("emotional_intensity"), Intensity, Intensity.low),
            "background_noise_present": present,
            "background_noise_type": str(d.get("background_noise_type") or ""),
            "background_noise_severity": _coerce(
                d.get("background_noise_severity"), NoiseSeverity, NoiseSeverity.none
            ),
            "audio_quality": _coerce(d.get("audio_quality"), AudioQuality, AudioQuality.clear),
            "speaker_overlap_present": bool(d.get("speaker_overlap_present", False)),
            "long_silence_present": bool(d.get("long_silence_present", False)),
            "confidence": conf,
        }
        if not present:
            out["background_noise_type"] = ""
            out["background_noise_severity"] = NoiseSeverity.none.value
        return cls(**out)
