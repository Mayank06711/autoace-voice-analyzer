"""OpenAI audio-capable emotion provider (optional; needs OPENAI_API_KEY)."""
from __future__ import annotations

import asyncio
import base64
import io
import json

import soundfile as sf

from app.analysis.emotion.base import EmotionProvider, EmotionResult
from app.analysis.emotion.pricing import cost_usd
from app.analysis.emotion.prompt import SYSTEM
from app.errors import ProviderError

USER = "Analyze this customer-support call audio and return the JSON object as specified."


def _wav_b64(clip) -> str:
    buf = io.BytesIO()
    sf.write(buf, clip.samples, clip.sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OpenAIProvider(EmotionProvider):
    name = "openai"

    def __init__(self, settings, model: str | None = None):
        self.s = settings
        # explicit model (chain step) wins; else take one only from the openai family (so a
        # Gemini-primary chain doesn't hand this provider a "gemini-*" model); else default.
        m = model or settings.openai_model or settings.emotion_model or ""
        self.model = m if m.startswith("gpt") else "gpt-audio-mini"
        self._client = None
        if settings.openai_api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.api_timeout_s)

    async def apredict(self, clip, context) -> EmotionResult:
        if self._client is None:
            raise ProviderError("OPENAI_API_KEY not set")
        return await asyncio.get_event_loop().run_in_executor(None, self._call, clip)

    def _call(self, clip) -> EmotionResult:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                modalities=["text"],
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER},
                        {"type": "input_audio",
                         "input_audio": {"data": _wav_b64(clip), "format": "wav"}},
                    ]},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
        except Exception as e:  # network / API error → let pipeline fall back
            raise ProviderError(f"openai call failed: {e}") from e
        result = _parse(raw)
        _attach_usage(result, self.model, resp)
        return result

    def disclose(self) -> dict:
        # metered per-token (audio input billed separately & far above text). Real $ is computed
        # per call from usage; here we surface the rates for the cost report.
        from app.analysis.emotion.pricing import rates
        a, ti, to = rates(self.model)
        return {"model": self.model, "audio_in_per_1m": a, "text_in_per_1m": ti,
                "text_out_per_1m": to, "audio_egress": True}


def _attach_usage(result: EmotionResult, model: str, resp) -> None:
    """Record which model ran + real token counts + computed $ onto the result (for the dashboard)."""
    u = getattr(resp, "usage", None)
    audio = text_in = out = 0
    if u is not None:
        det = getattr(u, "prompt_tokens_details", None)
        audio = int(getattr(det, "audio_tokens", 0) or 0) if det else 0
        text_in = int((u.prompt_tokens or 0) - audio)
        out = int(u.completion_tokens or 0)
    result.provider = "openai"
    result.model = model
    result.audio_tokens = audio
    result.text_in_tokens = text_in
    result.text_out_tokens = out
    result.cost_usd = cost_usd(model, audio, text_in, out)


def _parse(raw: str) -> EmotionResult:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ProviderError(f"no JSON in provider response: {raw[:120]}")
    data = json.loads(raw[start:end + 1])
    from app.domain.schema import EmotionalTone, Intensity

    def pick(v, enum, default):
        try:
            return enum(v)
        except Exception:
            return default

    return EmotionResult(
        tone=pick(data.get("emotional_tone"), EmotionalTone, EmotionalTone.neutral),
        intensity=pick(data.get("emotional_intensity"), Intensity, Intensity.low),
        confidence=float(data.get("confidence", 0.5)),
        overlap=bool(data["speaker_overlap_present"]) if "speaker_overlap_present" in data else None,
        noise_type=(data.get("background_noise_type") or None),
    )
