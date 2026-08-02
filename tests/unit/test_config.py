from app.config import Settings, get_settings


def test_defaults_are_sane():
    s = Settings()
    assert s.sample_rate == 16000
    assert s.max_workers >= 1
    assert 0 < s.port < 65536
    assert s.storage_backend == "local"
    assert s.vad_backend in {"webrtc", "silero"}


def test_get_settings_is_cached_singleton():
    assert get_settings() is get_settings()
