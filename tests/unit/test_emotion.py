import asyncio

import numpy as np
import pytest

from app.analysis.emotion.providers.mock import MockProvider
from app.analysis.emotion.registry import available, get_provider
from app.config import get_settings
from app.domain.contracts import AudioClip, EmotionResult
from app.errors import ProviderError

S = get_settings()


def _clip():
    return AudioClip(samples=np.zeros(16000, np.float32), sr=16000, duration=1.0)


def test_mock_returns_valid_result():
    r = asyncio.run(MockProvider(S).apredict(_clip(), {}))
    assert isinstance(r, EmotionResult)
    assert 0.0 <= r.confidence <= 1.0


def test_registry_resolves_mock_and_lists():
    assert get_provider("mock", S).name == "mock"
    assert "mock" in available()


def test_registry_unknown_raises():
    with pytest.raises(ProviderError):
        get_provider("nope", S)
