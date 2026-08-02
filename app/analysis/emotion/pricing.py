"""Token pricing for emotion providers → real per-call cost shown on the dashboard.

Rates are USD per 1,000,000 tokens, taken from each vendor's official pricing page (Aug 2026).
Audio-input is the dominant cost driver for call audio (OpenAI bills it far above text). Cost is
computed from the ACTUAL token counts each provider returns, not an estimate — so the dashboard shows
tokens-used × real rate. Unknown models fall back to a zero-rate (cost shown as $0, never crashes).
"""
from __future__ import annotations

# {model: (audio_in, text_in, text_out)} in USD per 1M tokens.
PRICES: dict[str, tuple[float, float, float]] = {
    # OpenAI audio models — audio input is separate & much higher than text.
    "gpt-audio-mini": (10.0, 0.60, 2.40),
    "gpt-audio": (32.0, 2.50, 10.0),
    "gpt-4o-audio-preview": (40.0, 2.50, 10.0),
    "gpt-4o-mini-audio-preview": (10.0, 0.15, 0.60),
    # Google Gemini — audio billed at a modest premium over text.
    "gemini-2.5-flash-lite": (0.30, 0.10, 0.40),
    "gemini-2.5-flash": (1.00, 0.30, 2.50),
    # Gemini 3.x — the flash tiers HEAR subtle noise (tested: 'radio'/'static') but the full
    # flash is over the $0.003/min ceiling ($1.50/M input); the lite tiers stay cheap.
    "gemini-3.6-flash": (1.50, 1.50, 7.50),
    "gemini-3.5-flash": (1.00, 1.00, 4.00),
    "gemini-3-flash-preview": (1.00, 0.50, 3.00),
    "gemini-3.5-flash-lite": (0.30, 0.10, 0.40),
    "gemini-3.1-flash-lite": (0.30, 0.10, 0.40),
    # local / no-cost
    "mock": (0.0, 0.0, 0.0),
    "disabled": (0.0, 0.0, 0.0),
}


def rates(model: str) -> tuple[float, float, float]:
    """Per-1M (audio_in, text_in, text_out) for a model; 0s if unknown (prefix match tolerated)."""
    if model in PRICES:
        return PRICES[model]
    for key, val in PRICES.items():  # tolerate suffixes like "-2026-01-19"
        if model.startswith(key):
            return val
    return (0.0, 0.0, 0.0)


def cost_usd(model: str, audio_tokens: int, text_in_tokens: int, text_out_tokens: int) -> float:
    """Exact call cost from real token counts × the model's per-token rates."""
    a, ti, to = rates(model)
    cost = audio_tokens * a / 1e6 + text_in_tokens * ti / 1e6 + text_out_tokens * to / 1e6
    return round(cost, 6)


def rates_dict(model: str) -> dict:
    """Named per-1M rates for a model — used to show the cost BASIS on the dashboard."""
    a, ti, to = rates(model)
    return {"audio_in": a, "text_in": ti, "text_out": to}
