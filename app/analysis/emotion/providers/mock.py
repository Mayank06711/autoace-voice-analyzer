"""Mock + disabled providers — let the whole pipeline run with no API key (build/test)."""
from __future__ import annotations

import numpy as np

from app.analysis.emotion.base import EmotionProvider, EmotionResult
from app.domain.schema import EmotionalTone, Intensity

_TONES = list(EmotionalTone)


class MockProvider(EmotionProvider):
    """Deterministic pseudo-emotion derived from simple audio stats (stable, no network)."""

    name = "mock"

    def __init__(self, settings=None, model: str | None = None):
        self.settings = settings

    async def apredict(self, clip, context) -> EmotionResult:
        y = clip.samples
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-9))
        idx = int((rms * 1000)) % len(_TONES)
        intensity = (
            Intensity.high if rms > 0.2 else Intensity.medium if rms > 0.05 else Intensity.low
        )
        return EmotionResult(tone=_TONES[idx], intensity=intensity, confidence=0.5, overlap=None,
                             provider="mock", model="mock")

    def disclose(self) -> dict:
        return {"model": "mock", "price_per_min": 0.0, "audio_egress": False}


class DisabledProvider(EmotionProvider):
    """Emotion disabled: safe neutral default at low confidence (acoustic fields still produced)."""

    name = "disabled"

    def __init__(self, settings=None, model: str | None = None):
        self.settings = settings

    async def apredict(self, clip, context) -> EmotionResult:
        return EmotionResult(
            tone=EmotionalTone.neutral, intensity=Intensity.low, confidence=0.2, overlap=None,
            provider="disabled", model="disabled",
        )

    def disclose(self) -> dict:
        return {"model": "disabled", "price_per_min": 0.0, "audio_egress": False}
