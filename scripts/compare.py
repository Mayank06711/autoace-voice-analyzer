"""Compare HYBRID (DSP + Gemini emotion) vs ALL-LLM (Gemini all 9), and show the confidence breakdown.

Both use structured output (response_schema). Key from env.
Run: GEMINI_API_KEY=... python scripts/compare.py
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
from google import genai
from google.genai import types

from app.analysis import merge
from app.analysis.emotion.prompt import SYSTEM
from app.analysis.emotion.registry import get_provider
from app.analysis.pipeline import run_acoustic
from app.audio.features import extract_features
from app.audio.io import decode
from app.config import get_settings
from app.domain.schema import AnalysisResult

CALLS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]

ALL9 = """\
<role>You are a senior call-center QA analyst and speech/acoustic specialist.</role>
<task>Analyze the CUSTOMER in this support-call recording and fill every field.</task>
<rules>
- Judge emotion from meaning + prosody, not loudness. Assess the customer, not the agent.
- audio_quality is technical fidelity ONLY (clipping/muffle/distortion) — independent of background noise.
- background_noise_present = meaningful non-speech sound audible; type = short description or "".
- confidence 0-1: certainty in the overall result; lower for short/ambiguous clips.
</rules>
Return JSON matching the schema.
"""

VIEW = ["emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
        "background_noise_severity", "audio_quality", "speaker_overlap_present", "long_silence_present",
        "confidence"]


def _wav(clip):
    buf = io.BytesIO()
    sf.write(buf, clip.samples, clip.sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def all_llm(client, model, clip):
    resp = client.models.generate_content(
        model=model,
        contents=[ALL9, types.Part.from_bytes(data=_wav(clip), mime_type="audio/wav")],
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=AnalysisResult, temperature=0.2),
    )
    return AnalysisResult.repair(json.loads(resp.text)).model_dump(mode="json")


async def main():
    s = get_settings()
    provider = get_provider("gemini", s)
    client = genai.Client(api_key=s.gemini_api_key)
    model = s.emotion_model or "gemini-2.5-flash-lite"

    for n in CALLS:
        clip = decode(Path("data") / n, s)
        feats = extract_features(clip, s)
        aco = run_acoustic(feats, clip.samples, s)
        emo = await provider.apredict(clip, {})
        ac_cert = merge.acoustic_confidence(feats, aco["audio_quality"])
        method = merge.resolve_overlap_method(s, provider)
        hyb = merge.combine(aco, emo, s, method, acoustic_conf=ac_cert).model_dump(mode="json")
        allm = all_llm(client, model, clip)

        print(f"==== {n} ====")
        print(f"  confidence: emotion_conf={emo.confidence:.2f}  quality={aco['audio_quality']}  "
              f"speech_frac={feats.is_speech.mean():.2f}  acoustic_certainty={ac_cert:.2f}  "
              f"=> overall={hyb['confidence']:.2f}")
        print("  HYBRID :", {k: hyb[k] for k in VIEW})
        print("  ALL-LLM:", {k: allm[k] for k in VIEW})


if __name__ == "__main__":
    asyncio.run(main())
