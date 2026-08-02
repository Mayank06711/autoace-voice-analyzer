"""Gemini audio emotion provider (optional; needs GEMINI_API_KEY).

Uses Gemini structured output (response_schema=EmotionOut) → guaranteed valid JSON, no text parsing.
Gemini 2.5 Flash-Lite audio ≈ $0.0006/audio-min — the recommended cost-compliant provider.
"""
from __future__ import annotations

import asyncio
import io
import json

import soundfile as sf

from app.analysis.emotion.base import EmotionProvider, EmotionResult
from app.analysis.emotion.pricing import cost_usd
from app.analysis.emotion.prompt import CUSTOMER_DESCRIBE, SYSTEM, EmotionOut
from app.domain.schema import EmotionalTone, Intensity
from app.errors import ProviderError


def _wav_bytes(clip) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, clip.samples, clip.sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class GeminiProvider(EmotionProvider):
    name = "gemini"

    def __init__(self, settings, model: str | None = None):
        self.s = settings
        # explicit model (chain step) wins; else take one only from the gemini family (so an
        # openai-primary chain doesn't hand this provider a "gpt-*" model); else default.
        m = model or settings.gemini_model or settings.emotion_model or ""
        self.model = m if m.startswith("gemini") else "gemini-2.5-flash-lite"
        self._client = None
        if settings.gemini_api_key:
            from google import genai

            self._client = genai.Client(api_key=settings.gemini_api_key)

    async def apredict(self, clip, context) -> EmotionResult:
        if self._client is None:
            raise ProviderError("GEMINI_API_KEY not set")
        return await asyncio.get_event_loop().run_in_executor(None, self._call, clip)

    def _call(self, clip) -> EmotionResult:
        from google.genai import types

        audio = types.Part.from_bytes(data=_wav_bytes(clip), mime_type="audio/wav")
        tok = {"audio": 0, "text_in": 0, "text_out": 0}
        try:
            contents = [SYSTEM, audio]
            if self.s.emotion_two_pass:  # pass 1: isolate + describe the customer
                pass1 = self._client.models.generate_content(
                    model=self.model, contents=[CUSTOMER_DESCRIBE, audio])
                _add_usage(tok, pass1)
                notes = pass1.text or ""
                contents = [SYSTEM, audio, "Analyst notes on the customer: " + notes.strip()]
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EmotionOut,
                    temperature=0.2,
                ),
            )
            _add_usage(tok, resp)
            raw = resp.text or "{}"
        except Exception as e:
            raise ProviderError(f"gemini call failed: {e}") from e

        d = json.loads(raw)
        return EmotionResult(
            tone=EmotionalTone(d["emotional_tone"]),
            intensity=Intensity(d["emotional_intensity"]),
            confidence=float(d.get("confidence", 0.5)),
            overlap=bool(d.get("speaker_overlap_present", False)),
            noise_type=(d.get("background_noise_type") or None),
            provider="gemini", model=self.model,
            audio_tokens=tok["audio"], text_in_tokens=tok["text_in"], text_out_tokens=tok["text_out"],
            cost_usd=cost_usd(self.model, tok["audio"], tok["text_in"], tok["text_out"]),
        )

    def disclose(self) -> dict:
        # metered per-token; real $ computed per call from usage_metadata. Surface rates for reporting.
        from app.analysis.emotion.pricing import rates
        a, ti, to = rates(self.model)
        return {"model": self.model, "audio_in_per_1m": a, "text_in_per_1m": ti,
                "text_out_per_1m": to, "audio_egress": True}


def _add_usage(tok: dict, resp) -> None:
    """Accumulate audio/text token counts from a Gemini response's usage_metadata (per-modality)."""
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return
    tok["text_out"] += int(getattr(um, "candidates_token_count", 0) or 0)
    details = getattr(um, "prompt_tokens_details", None)
    if details:  # split prompt tokens by modality (AUDIO vs TEXT)
        for d in details:
            mod = str(getattr(d, "modality", "")).upper()
            n = int(getattr(d, "token_count", 0) or 0)
            if "AUDIO" in mod:
                tok["audio"] += n
            else:
                tok["text_in"] += n
    else:  # no breakdown → count all prompt tokens as audio (the dominant, higher-priced modality)
        tok["audio"] += int(getattr(um, "prompt_token_count", 0) or 0)
