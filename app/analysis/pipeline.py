"""Analysis pipeline: run_acoustic (6 fields) and run_file (full 9 fields, Track A ∥ Track B)."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.analysis import merge
from app.analysis.acoustic import noise, noise_type, overlap, quality, silence
from app.audio.features import extract_features
from app.audio.io import decode
from app.config import Settings, get_settings
from app.domain.contracts import AudioClip, EmotionProvider, EmotionResult, Features
from app.domain.schema import AnalysisResult, EmotionalTone, Intensity
from app.errors import ProviderError
from app.logging_conf import event

logger = logging.getLogger(__name__)

# Registry of acoustic detectors. Adding one = new module + one line here.
ACOUSTIC = [silence.detect, noise.detect, quality.detect, noise_type.detect, overlap.detect]

# Shared bounded thread pool for CPU-bound DSP (decode/features/detectors), so the async event
# loop is NEVER blocked. Sized to APP_MAX_WORKERS; created once.
_EXECUTOR: ThreadPoolExecutor | None = None


def get_executor(s: Settings) -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, s.max_workers), thread_name_prefix="dsp")
    return _EXECUTOR


def run_acoustic(features: Features, audio: np.ndarray, s: Settings) -> dict:
    """Run every acoustic detector and merge their field-dicts into the 6 acoustic fields."""
    out: dict = {}
    for detect in ACOUSTIC:
        out.update(detect(features, audio, s))
    return out


def _is_quota_error(e: Exception) -> bool:
    """A 429 / quota / rate-limit won't recover in seconds → skip to the next model immediately
    rather than burning the backoff budget retrying the same exhausted model."""
    m = str(e).lower()
    return any(k in m for k in
               ("429", "resource_exhausted", "quota", "rate limit", "rate_limit", "insufficient_quota"))


async def _safe_emotion(providers, clip: AudioClip, s: Settings) -> EmotionResult:
    """Track B: try each provider/model in the chain, cascading on failure.

    A transient error retries the SAME model with jittered backoff; a quota/429 (or exhausted
    retries, or a non-transient error) moves on to the next model in the chain — e.g.
    gemini-3.6-flash → gemini-3.5-flash-lite → gemini-2.5-flash-lite → gpt-audio-mini. If the whole
    chain fails → safe partial result (low confidence), so a file is never lost.
    """
    last: Exception | None = None
    for provider in providers:
        delay = s.retry_base_s
        for attempt in range(s.retry_max + 1):
            try:
                return await asyncio.wait_for(provider.apredict(clip, {}), timeout=s.api_timeout_s)
            except (ProviderError, asyncio.TimeoutError) as e:
                last = e
                if _is_quota_error(e) or attempt >= s.retry_max:
                    break  # exhausted model or spent retries → cascade to the next model
                await asyncio.sleep(random.uniform(0, min(delay, s.retry_cap_s)))
                delay = min(delay * 2, s.retry_cap_s)
            except Exception as e:  # non-retryable → skip to the next model immediately
                last = e
                break
        event(logger, logging.WARNING, "provider.exhausted",
              provider=getattr(provider, "name", "?"), model=getattr(provider, "model", ""),
              reason=str(last))
    event(logger, logging.WARNING, "file.partial", reason=str(last))
    return EmotionResult(
        tone=EmotionalTone.neutral, intensity=Intensity.low, confidence=0.2, overlap=None
    )


async def run_file(path, providers, s: Settings | None = None) -> tuple[AnalysisResult, dict]:
    """Decode → features → (acoustic on threads ∥ emotion async) → merge → (result, cost_meta).

    ALL blocking DSP (decode/features/detectors) runs in the bounded thread pool, so the single
    uvicorn event loop is never blocked; emotion (I/O) runs concurrently. `providers` may be a single
    EmotionProvider or an ordered fallback chain (see build_chain). The second return value is cost
    telemetry (provider/model/tokens/$), kept OUT of the immutable 9-field result.
    """
    s = s or get_settings()
    providers = list(providers) if isinstance(providers, (list, tuple)) else [providers]
    started = time.perf_counter()
    loop = asyncio.get_event_loop()
    pool = get_executor(s)

    clip = await loop.run_in_executor(pool, decode, path, s)  # off the event loop

    async def _acoustic():
        features = await loop.run_in_executor(pool, extract_features, clip, s)
        aco = await loop.run_in_executor(pool, run_acoustic, features, clip.samples, s)
        return features, aco

    (features, acoustic), emotion = await asyncio.gather(
        _acoustic(), _safe_emotion(providers, clip, s)
    )

    method = merge.resolve_overlap_method(s, providers[0])
    ac_conf = merge.acoustic_confidence(features, acoustic.get("audio_quality", "clear"))
    result = merge.combine(acoustic, emotion, s, method, acoustic_conf=ac_conf)
    meta = {
        "provider": emotion.provider, "model": emotion.model,
        "audio_tokens": emotion.audio_tokens, "text_in_tokens": emotion.text_in_tokens,
        "text_out_tokens": emotion.text_out_tokens, "cost_usd": emotion.cost_usd,
        "audio_seconds": round(float(clip.duration), 2),
    }
    event(logger, logging.DEBUG, "file.analyzed", ms=int((time.perf_counter() - started) * 1000))
    return result, meta
