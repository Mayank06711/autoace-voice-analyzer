"""merge.py — the arbiter. Combines the 6 acoustic fields + emotion into a valid AnalysisResult.

Resolves cross-field rules that individual detectors deliberately don't: picks the overlap source
(provider in llm mode, else the heuristic detector) and, via AnalysisResult.repair, clears
background_noise_type when noise isn't present, coerces enums, and clamps confidence.
"""
from __future__ import annotations

from app.domain.contracts import EmotionResult, Features
from app.domain.schema import AnalysisResult

# Overall confidence semantics (documented): it is NOT just the LLM's number. The LLM only reports
# certainty about the 3 EMOTION fields; the 6 acoustic fields are deterministic but only trustworthy
# when the audio is actually analyzable. So:
#     confidence = W_EMOTION * emotion_conf  +  W_ACOUSTIC * acoustic_certainty
#     acoustic_certainty = quality_factor(audio_quality) * (0.5 + 0.5 * speech_factor)
#     speech_factor = min(1, speech_fraction / 0.30)
# => 1.0 only when the LLM is fully sure AND audio is clear AND there's ample speech.
# => ~0.5 when one source is uncertain (ambiguous emotion, or impaired/low-speech audio).
_W_EMOTION = 0.6
_W_ACOUSTIC = 0.4
_QUALITY_FACTOR = {"clear": 1.0, "slightly_impaired": 0.7, "severely_impaired": 0.4}

# Values a provider may return that actually mean "no noise" — never let these become a noise TYPE
# (contradicts background_noise_present=True). We fall back to the DSP label instead.
_NO_NOISE_WORDS = {"", "none", "n/a", "na", "silence", "silent", "quiet", "no noise", "unknown"}


def resolve_overlap_method(s, provider) -> str:
    """`auto` → llm when the provider is an audio-LLM, else heuristic."""
    method = (s.overlap_method or "auto").lower()
    if method != "auto":
        return method
    return "llm" if getattr(provider, "name", "") in {"openai", "gemini", "anthropic"} else "heuristic"


def acoustic_confidence(features: Features | None, audio_quality: str) -> float:
    """Certainty of the DETERMINISTIC half: high when audio is clean and there's enough speech."""
    qf = _QUALITY_FACTOR.get(audio_quality, 0.7)
    speech_frac = float(features.is_speech.mean()) if features is not None else 0.5
    speech_factor = min(1.0, speech_frac / 0.3)  # too little speech → less certain
    return round(qf * (0.5 + 0.5 * speech_factor), 3)


def combine(acoustic: dict, emotion: EmotionResult, s, overlap_method: str,
            acoustic_conf: float | None = None) -> AnalysisResult:
    data = dict(acoustic)
    data["emotional_tone"] = emotion.tone.value
    data["emotional_intensity"] = emotion.intensity.value

    if overlap_method == "llm" and emotion.overlap is not None:
        data["speaker_overlap_present"] = bool(emotion.overlap)
    # else keep the acoustic heuristic's speaker_overlap_present

    # noise_type: DSP owns presence/severity; the LLM gives a better description WHEN it hears one.
    # Prefer the LLM's type only if DSP confirms noise AND the LLM returned a real descriptor (not a
    # "no-noise" sentinel like "none"/"silence" — those would contradict present=True). Otherwise the
    # DSP heuristic's label stands. repair() clears the field to "" when noise isn't present.
    llm_type = (emotion.noise_type or "").strip()
    if data.get("background_noise_present") and llm_type.lower() not in _NO_NOISE_WORDS:
        data["background_noise_type"] = llm_type

    # blended overall confidence: the LLM is uncertain (emotion), the DSP is deterministic (acoustic)
    ac = acoustic_conf if acoustic_conf is not None else 0.85
    data["confidence"] = round(_W_EMOTION * float(emotion.confidence) + _W_ACOUSTIC * ac, 2)

    return AnalysisResult.repair(data)
