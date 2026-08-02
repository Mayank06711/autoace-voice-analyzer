import numpy as np

from app.analysis.acoustic import noise, quality, silence
from app.audio.features import extract_features
from app.config import get_settings
from app.domain.contracts import AudioClip
from tests.fixtures.signals import clipped_tone, concat
from tests.fixtures.signals import silence as sil
from tests.fixtures.signals import tone

S = get_settings()


def _feat(y, sr=16000):
    clip = AudioClip(samples=y.astype(np.float32), sr=sr, duration=len(y) / sr)
    return extract_features(clip, S), clip.samples


def test_long_silence_fires_on_long_gap():
    f, a = _feat(concat(tone(dur=1.0), sil(dur=6.0), tone(dur=1.0)))
    assert silence.detect(f, a, S)["long_silence_present"] is True


def test_long_silence_absent_on_continuous_tone():
    f, a = _feat(tone(dur=3.0))
    assert silence.detect(f, a, S)["long_silence_present"] is False


def test_noise_absent_on_clean_silence():
    f, a = _feat(sil(dur=2.0))
    r = noise.detect(f, a, S)
    assert r["background_noise_present"] is False
    assert r["background_noise_severity"] == "none"


def test_noise_present_on_noisy_real_call():
    # call_002 has audible TV background — WADA on speech must flag it (noise-under-speech).
    from pathlib import Path

    from app.audio.io import decode

    clip = decode(Path("data") / "call_002.ogg", S)
    f = extract_features(clip, S)
    assert noise.detect(f, clip.samples, S)["background_noise_present"] is True


def test_quality_severe_on_heavy_clipping():
    f, a = _feat(clipped_tone(dur=1.0))
    assert quality.detect(f, a, S)["audio_quality"] == "severely_impaired"
