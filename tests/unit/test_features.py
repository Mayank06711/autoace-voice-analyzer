from pathlib import Path

import numpy as np

from app.audio.features import extract_features
from app.audio.io import decode
from app.domain.contracts import AudioClip
from tests.fixtures.signals import silence, tone, white_noise


def _clip(y, sr=16000):
    return AudioClip(samples=y.astype(np.float32), sr=sr, duration=len(y) / sr)


def test_alignment_all_timelines_equal_length():
    f = extract_features(_clip(tone(dur=2.0)))
    t = len(f.energy_dbfs)
    assert len(f.is_speech) == t
    for v in f.spectral.values():
        assert len(v) == t


def test_energy_tracks_amplitude():
    loud = extract_features(_clip(tone(dur=1.0, amp=0.8)))
    quiet = extract_features(_clip(tone(dur=1.0, amp=0.05)))
    assert np.median(loud.energy_dbfs) > np.median(quiet.energy_dbfs)


def test_silence_is_low_energy():
    f = extract_features(_clip(silence(dur=1.0)))
    assert np.median(f.energy_dbfs) < -60


def test_spectrum_flatter_for_noise_than_tone():
    n = extract_features(_clip(white_noise(dur=1.0, amp=0.2)))
    t = extract_features(_clip(tone(freq=300, dur=1.0)))
    assert np.median(n.spectral["flatness"]) > np.median(t.spectral["flatness"])


def test_vad_flags_speech_on_real_call():
    # VAD is trained on speech; a synthetic tone won't fire, so use a real call.
    clip = decode(Path("data") / "call_001.ogg")
    f = extract_features(clip)
    assert f.is_speech.mean() > 0.1
