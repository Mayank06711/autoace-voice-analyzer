from app.domain.schema import AnalysisResult, EmotionalTone, NoiseSeverity

EXAMPLE = {
    "emotional_tone": "frustrated",
    "emotional_intensity": "medium",
    "background_noise_present": True,
    "background_noise_type": "office chatter",
    "background_noise_severity": "low",
    "audio_quality": "clear",
    "speaker_overlap_present": False,
    "long_silence_present": False,
    "confidence": 0.82,
}


def test_valid_example_validates():
    r = AnalysisResult(**EXAMPLE)
    assert r.emotional_tone is EmotionalTone.frustrated
    assert r.confidence == 0.82


def test_repair_bad_enum_and_clamps_confidence():
    r = AnalysisResult.repair({**EXAMPLE, "emotional_tone": "angry", "confidence": 1.5})
    assert r.emotional_tone is EmotionalTone.neutral  # unknown -> safe default
    assert r.confidence == 1.0                         # clamped into [0,1]


def test_repair_no_noise_clears_type_and_severity():
    r = AnalysisResult.repair(
        {**EXAMPLE, "background_noise_present": False, "background_noise_type": "tv",
         "background_noise_severity": "high"}
    )
    assert r.background_noise_type == ""
    assert r.background_noise_severity is NoiseSeverity.none


def test_repair_handles_empty_dict():
    r = AnalysisResult.repair({})
    assert r.emotional_tone is EmotionalTone.neutral
    assert 0.0 <= r.confidence <= 1.0


def test_repair_non_numeric_confidence_never_crashes():
    # a malformed provider/manifest value must degrade, not raise (never-crash contract)
    for bad in ["high", None, {}, "0.8x"]:
        r = AnalysisResult.repair({**EXAMPLE, "confidence": bad})
        assert 0.0 <= r.confidence <= 1.0
