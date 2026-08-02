"""Shared data objects and interfaces used across layers (no framework imports here)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from app.domain.schema import EmotionalTone, Intensity


@dataclass(frozen=True)
class AudioClip:
    """Decoded audio: mono float32 at a known sample rate. Produced by audio/io.decode()."""

    samples: np.ndarray
    sr: int
    duration: float


@dataclass(frozen=True)
class Features:
    """Frame-aligned timelines. energy_dbfs, is_speech and each spectral[...] share one length."""

    sr: int
    hop_ms: int
    energy_dbfs: np.ndarray
    is_speech: np.ndarray
    spectral: dict
    duration_s: float


class EmotionResult(BaseModel):
    """Track B output. `overlap` is populated only when overlap is judged by the LLM call.

    The trailing fields are cost/usage TELEMETRY (which provider actually ran + tokens billed + the
    computed $). They are metadata for the dashboard/cost report, NOT part of the immutable 9-field
    schema, so they never enter AnalysisResult / result_json.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tone: EmotionalTone
    intensity: Intensity
    confidence: float
    overlap: Optional[bool] = None
    noise_type: Optional[str] = None  # LLM's background-noise description (used only if DSP says present)

    # --- cost/usage telemetry (defaults = local/no-cost providers) ---
    provider: str = ""                 # provider that produced THIS result (after any fallback)
    model: str = ""                    # concrete model name, for pricing + display
    audio_tokens: int = 0
    text_in_tokens: int = 0
    text_out_tokens: int = 0
    cost_usd: float = 0.0


class EmotionProvider(ABC):
    """Strategy interface for emotion. Implementations live in analysis/emotion/providers/."""

    name: str = "base"

    @abstractmethod
    async def apredict(self, clip: AudioClip, context: dict) -> EmotionResult:
        """Return the emotional fields for a clip (async so I/O calls run concurrently)."""

    def disclose(self) -> dict:
        """Cost/privacy disclosure for the memo: model, price/min, whether audio leaves infra."""
        return {"model": self.name, "price_per_min": 0.0, "audio_egress": False}


# Acoustic detectors are plain callables with the signature:
#     detect(features: Features, audio: np.ndarray, settings) -> dict
# (kept as a convention, not an ABC, so they stay pure functions.)
