"""Provider factory: name -> class. Add a vendor = new module + one registry line."""
from __future__ import annotations

import logging

from app.analysis.emotion.base import EmotionProvider
from app.analysis.emotion.providers.mock import DisabledProvider, MockProvider
from app.config import Settings, get_settings
from app.errors import ProviderError

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[EmotionProvider]] = {
    "mock": MockProvider,
    "disabled": DisabledProvider,
}


def _register_optional() -> None:
    """Import real providers lazily so a missing SDK never breaks startup."""
    try:
        from app.analysis.emotion.providers.openai import OpenAIProvider

        _REGISTRY["openai"] = OpenAIProvider
    except Exception:  # pragma: no cover
        logger.debug("openai provider unavailable")
    try:
        from app.analysis.emotion.providers.gemini import GeminiProvider

        _REGISTRY["gemini"] = GeminiProvider
    except Exception:  # pragma: no cover
        logger.debug("gemini provider unavailable")


_register_optional()


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str | None = None, s: Settings | None = None,
                 mode: str | None = None, model: str | None = None) -> EmotionProvider:
    s = s or get_settings()
    if (mode or s.emotion_mode) == "disabled":
        return DisabledProvider(s)
    key = (name or s.emotion_provider or "mock").lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ProviderError(f"unknown emotion provider: {key!r} (have {available()})")
    return cls(s, model=model)


def build_chain(name: str | None = None, s: Settings | None = None,
                mode: str | None = None) -> list[EmotionProvider]:
    """Ordered provider/model cascade tried best→cheapest→cross-provider.

    If APP_EMOTION_CHAIN is set it wins — a comma list of `provider:model` steps, e.g.
    `gemini:gemini-3.6-flash,gemini:gemini-3.5-flash-lite,gemini:gemini-2.5-flash-lite,openai:gpt-audio-mini`.
    The pipeline tries each in order, so a 429/quota on the best model degrades to a cheaper (or
    cross-provider) one. Falls back to the single primary+fallback form when no chain is configured.
    """
    s = s or get_settings()
    if (mode or s.emotion_mode) == "disabled":
        return [DisabledProvider(s)]

    cfg = (s.emotion_chain or "").strip()
    if cfg:
        chain: list[EmotionProvider] = []
        for step in cfg.split(","):
            step = step.strip()
            if not step:
                continue
            pname, _, pmodel = step.partition(":")
            pname = pname.strip().lower()
            if pname not in _REGISTRY:
                logger.debug("chain step %r: unknown provider", step)
                continue
            try:
                chain.append(get_provider(pname, s, mode, model=(pmodel.strip() or None)))
            except Exception as e:  # a broken step must not sink the whole chain
                logger.debug("chain step %r unavailable: %s", step, e)
        if chain:
            return chain

    chain = [get_provider(name, s, mode)]
    fb = (s.emotion_fallback or "").lower()
    if fb and fb != chain[0].name and fb in _REGISTRY:
        try:
            chain.append(_REGISTRY[fb](s))
        except Exception as e:  # a broken fallback must never block the primary
            logger.debug("fallback provider %s unavailable: %s", fb, e)
    return chain
