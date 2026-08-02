"""merge.py arbiter — noise_type source selection and the no-noise sentinel guard."""
from __future__ import annotations

from app.analysis import merge
from app.config import get_settings
from app.domain.contracts import EmotionResult
from app.domain.schema import EmotionalTone, Intensity


def _emotion(noise_type: str) -> EmotionResult:
    return EmotionResult(tone=EmotionalTone.neutral, intensity=Intensity.low,
                         confidence=0.8, overlap=None, noise_type=noise_type)


def _combine(acoustic: dict, noise_type: str):
    s = get_settings()
    return merge.combine(acoustic, _emotion(noise_type), s, overlap_method="llm", acoustic_conf=0.9)


def test_llm_noise_type_used_when_real_descriptor():
    r = _combine({"background_noise_present": True, "background_noise_type": "ambient noise",
                  "background_noise_severity": "medium"}, "traffic")
    assert r.background_noise_type == "traffic"  # LLM's real description wins over the DSP guess


def test_llm_none_sentinel_falls_back_to_dsp_label():
    # gpt-audio-mini often returns "none"/"" — that must NOT become a noise TYPE when noise is present
    for sentinel in ("none", "", "  ", "silence", "N/A"):
        r = _combine({"background_noise_present": True, "background_noise_type": "static/hiss",
                      "background_noise_severity": "medium"}, sentinel)
        assert r.background_noise_type == "static/hiss", sentinel


def test_noise_type_cleared_when_absent():
    # repair() must blank the type when noise isn't present, regardless of what the LLM said
    r = _combine({"background_noise_present": False, "background_noise_type": "",
                  "background_noise_severity": "none"}, "traffic")
    assert r.background_noise_type == ""
