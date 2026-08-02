"""Emotion cost pricing + provider fallback chain."""
from __future__ import annotations

import asyncio

import numpy as np

from app.analysis.emotion import pricing
from app.analysis.emotion.registry import build_chain
from app.analysis.pipeline import _is_quota_error, _safe_emotion
from app.config import Settings
from app.domain.contracts import AudioClip, EmotionProvider, EmotionResult
from app.domain.schema import EmotionalTone, Intensity
from app.errors import ProviderError

# retry_max=0 → no backoff sleeps; fail over to the next provider immediately
FAST = Settings(retry_max=0, retry_base_s=0.0, retry_cap_s=0.0, api_timeout_s=5.0)
CLIP = AudioClip(samples=np.zeros(16000, np.float32), sr=16000, duration=1.0)


def test_pricing_from_real_measured_tokens():
    # gpt-audio-mini on our 35s call: 349 audio + 983 text-in + 74 out ≈ $0.0043 (audio-in $10/1M)
    assert 0.004 < pricing.cost_usd("gpt-audio-mini", 349, 983, 74) < 0.005
    # Gemini Flash-Lite is ~10x cheaper for a comparable call
    assert pricing.cost_usd("gemini-2.5-flash-lite", 1300, 720, 90) < 0.001
    # local providers are free; unknown models never crash (treated as $0)
    assert pricing.cost_usd("mock", 999, 999, 999) == 0.0
    assert pricing.cost_usd("some-future-model", 100, 100, 100) == 0.0


class _Boom(EmotionProvider):
    name = "boom"

    async def apredict(self, clip, context) -> EmotionResult:
        raise ProviderError("primary down")


class _Good(EmotionProvider):
    name = "good"

    async def apredict(self, clip, context) -> EmotionResult:
        return EmotionResult(tone=EmotionalTone.satisfied, intensity=Intensity.low,
                             confidence=0.9, provider="good", model="good-1")


def test_fallback_uses_second_provider_when_primary_fails():
    res = asyncio.run(_safe_emotion([_Boom(), _Good()], CLIP, FAST))
    assert res.tone == EmotionalTone.satisfied
    assert res.provider == "good"  # the result carries the provider that actually ran


def test_whole_chain_failing_yields_safe_partial():
    res = asyncio.run(_safe_emotion([_Boom(), _Boom()], CLIP, FAST))
    assert res.tone == EmotionalTone.neutral
    assert res.confidence <= 0.3
    assert res.provider == ""  # partial default = no provider ran


def test_quota_error_detection():
    assert _is_quota_error(ProviderError("gemini call failed: 429 RESOURCE_EXHAUSTED"))
    assert _is_quota_error(ProviderError("rate limit exceeded"))
    assert not _is_quota_error(ProviderError("connection reset"))


def test_emotion_chain_builds_ordered_provider_model_cascade():
    # APP_EMOTION_CHAIN parses into the exact ordered (provider, model) steps, best→fallback
    s = Settings(emotion_mode="api", emotion_fallback="",
                 emotion_chain="gemini:gemini-3.6-flash,gemini:gemini-2.5-flash-lite,openai:gpt-audio-mini")
    chain = build_chain(s=s)
    assert [(p.name, p.model) for p in chain] == [
        ("gemini", "gemini-3.6-flash"),
        ("gemini", "gemini-2.5-flash-lite"),
        ("openai", "gpt-audio-mini"),
    ]
