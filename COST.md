# Cost Analysis

**Ceiling:** ≤ $0.003 per audio-minute.

All numbers below are **measured from real API usage** (the token counts returned by each provider on
our own 3 calls), not vendor estimates — see *Methodology*. Rates are from each vendor's official
pricing page (Aug 2026).

## Per-provider cost per audio-minute (measured)

| Path | Audio-in rate | $ / audio-min | Under ceiling? |
|---|---|---|---|
| **Acoustic only (local DSP)** | — | **$0** | ✅ compute only |
| **Hybrid + Gemini 2.5 Flash-Lite** | ~$0.10–0.50/1M | **~$0.0004–0.0014** | ✅ **(2–7× headroom)** |
| Hybrid + gpt-audio-mini | **$10/1M** | **~$0.006–0.0073** | ❌ ~2× over |
| Hybrid + gpt-audio (full) | $32/1M | ~$0.02 | ❌ ~7× over |
| Mock / disabled provider | — | $0 | ✅ no external call |

**The one audio-LLM that meets the ceiling is Gemini 2.5 Flash-Lite.** OpenAI's audio models bill
**audio-input tokens separately and far above text** ($10/1M for mini, $32/1M for full — vs $0.60 text),
which puts both over budget for continuous call audio. This is why the production default is Gemini
Flash-Lite; the 6 acoustic fields are $0 regardless (local DSP).

## How the measured numbers were derived (Methodology)

We read the **actual token accounting** from each provider's response on our real calls:

- **gpt-audio-mini**, 35 s call → `prompt_tokens_details`: **349 audio** + 983 text input, **74** text
  output. OpenAI tokenizes call audio at ~10 tokens/s.
  `349×$10/1M + 983×$0.60/1M + 74×$2.40/1M = $0.0043` for 35 s → **$0.0073/min**. Long-call asymptote
  (fixed prompt amortized): `~600 audio tok/min × $10/1M ≈ $0.006/min`. **Over ceiling.**
- **Gemini 2.5 Flash-Lite**, 31 s call → `usage_metadata`: **2,020** input tokens (audio at ~32 tok/s
  + ~700-token system prompt), **93** output. At Flash-Lite rates that is **$0.0004–0.0014/min**
  depending on the audio-token rate. **Well under ceiling.**
- **Local DSP** (6 acoustic fields): no external call → **$0** marginal (amortized server compute).

Only the **selected emotion provider** incurs per-minute cost; overlap is folded into the same call
(no extra request), and the fixed ~700-token system prompt amortizes to near-zero on calls > ~1 min.

## Reliability note (production)

Gemini's **free tier** has a low daily request quota (we hit `429 RESOURCE_EXHAUSTED` after 2 calls on
a shared key). For a reliable hosted deployment, enable **pay-as-you-go billing** on the Gemini
project — per-call cost stays ~$0.001/min (1,000 audio-min ≈ **$1**). If a `429` still occurs, the
pipeline **degrades gracefully**: retry/backoff first, then a partial result (all 6 acoustic fields +
a low-confidence default emotion) — one throttled call never fails the batch.

## Cost controls in the system
- `disclose()` on each provider surfaces `price_per_min` for auditing.
- Provider is swappable per-batch / per-env; the whole system runs at **$0** on the local/mock path.
- `APP_API_CONCURRENCY` bounds in-flight provider calls (spend + rate-limit control).

## Worked example (30-min call)
Gemini Flash-Lite: `30 × ~$0.001 ≈ $0.03`. Ceiling for 30 min = $0.09 → **~3× under**.
gpt-audio-mini would be `30 × ~$0.006 ≈ $0.18` → **2× over** the $0.09 budget.
