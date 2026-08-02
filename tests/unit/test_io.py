from pathlib import Path

import pytest

from app.audio.io import SUPPORTED_EXTS, decode, is_supported
from app.errors import DecodeError
from tests.fixtures.signals import tone, write_wav

DATA = Path("data")


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_decode_real_ogg(name):
    clip = decode(DATA / name)
    assert clip.sr == 16000
    assert clip.samples.ndim == 1          # mono
    assert clip.duration > 1.0


def test_is_supported():
    assert is_supported("x.OGG") and is_supported("x.mp3")
    assert not is_supported("x.txt")
    assert ".opus" in SUPPORTED_EXTS


def test_reject_corrupt(tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(DecodeError):
        decode(bad)


def test_reject_unsupported(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_bytes(b"x")
    with pytest.raises(DecodeError):
        decode(f)


def test_synth_wav_roundtrip(tmp_path):
    p = write_wav(tmp_path / "t.wav", tone(dur=1.0))
    clip = decode(p)
    assert abs(clip.duration - 1.0) < 0.05
