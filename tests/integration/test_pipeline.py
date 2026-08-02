import asyncio
from pathlib import Path

from app.analysis.emotion.base import EmotionProvider, EmotionResult
from app.analysis.emotion.registry import get_provider
from app.analysis.pipeline import run_file
from app.config import get_settings
from app.domain.schema import AnalysisResult

S = get_settings()
NINE = {
    "emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
    "background_noise_severity", "audio_quality", "speaker_overlap_present",
    "long_silence_present", "confidence",
}


def test_full_pipeline_emits_nine_valid_fields():
    provider = get_provider("mock", S)
    for name in ["call_001.ogg", "call_002.ogg", "call_003.ogg"]:
        result, meta = asyncio.run(run_file(Path("data") / name, provider, S))
        assert isinstance(result, AnalysisResult)
        assert set(result.model_dump(mode="json")) == NINE
        assert meta["model"] == "mock" and meta["cost_usd"] == 0.0  # telemetry present, $0 for mock


class _BoomProvider(EmotionProvider):
    name = "boom"

    async def apredict(self, clip, context) -> EmotionResult:
        raise RuntimeError("provider exploded")


def test_emotion_failure_yields_partial_result():
    # Provider failure must NOT crash the file — acoustic fields survive, emotion is safe default.
    result, _ = asyncio.run(run_file(Path("data") / "call_001.ogg", _BoomProvider(), S))
    assert isinstance(result, AnalysisResult)
    assert result.emotional_tone.value == "neutral"
    # emotion failed (self-conf 0.2) but acoustic is solid → blended confidence is moderate, not high
    assert result.confidence < 0.6
