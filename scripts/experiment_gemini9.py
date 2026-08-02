"""Experiment: ask Gemini for ALL 9 fields at once, using structured output (response_schema).

This is the 'end-to-end LLM' approach (a materially different second approach) vs. our hybrid.
Run: GEMINI_API_KEY=... python scripts/experiment_gemini9.py
Key is read from env (never inline).
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
from google import genai
from google.genai import types

from app.audio.io import decode
from app.domain.schema import AnalysisResult

MODEL = os.environ.get("APP_EMOTION_MODEL", "gemini-flash-latest")

SYSTEM = """\
You are an expert call-center audio analyst. Analyze the CUSTOMER in this support-call recording and
fill EVERY field. Judge emotion from wording + prosody, NOT loudness. Keep audio_quality independent
of background noise (a clear voice with a loud TV is still 'clear').

- emotional_tone: neutral | satisfied | frustrated | upset | distressed (customer's primary tone).
- emotional_intensity: low | medium | high.
- background_noise_present: is meaningful non-speech sound audible?
- background_noise_type: short description (e.g. TV, music, static, office chatter); "" if none.
- background_noise_severity: none | low | medium | high (how much it interferes).
- audio_quality: clear | slightly_impaired | severely_impaired (distortion/clipping/muffle only).
- speaker_overlap_present: do two people talk at once enough to affect understanding?
- long_silence_present: any unusually long dead-air gap?
- confidence: 0..1 — your certainty in the overall result (lower for short/ambiguous clips).
"""


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("set GEMINI_API_KEY")
        return 1
    client = genai.Client(api_key=key)
    out = {}
    for name in ["call_001.ogg", "call_002.ogg", "call_003.ogg"]:
        clip = decode(Path("data") / name)
        buf = io.BytesIO()
        sf.write(buf, clip.samples, clip.sr, format="WAV", subtype="PCM_16")
        resp = client.models.generate_content(
            model=MODEL,
            contents=[SYSTEM, types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav")],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,  # forces valid 9-field JSON
            ),
        )
        out[name] = AnalysisResult.repair(json.loads(resp.text)).model_dump(mode="json")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
