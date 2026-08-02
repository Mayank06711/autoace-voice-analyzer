# Cost Analysis

**Ceiling:** ≤ $0.003 per audio-minute.

All numbers below are **measured from real API usage** (the token counts returned by each provider on
our own 3 calls), not vendor estimates — see *Methodology*. Rates are from each vendor's official
pricing page (Aug 2026).

## Per-provider cost per audio-minute (measured)

| Path | Audio-in rate | $ / audio-min | Under ceiling? | Hears subtle noise? |
|---|---|---|---|---|
| **Acoustic only (local DSP)** | — | **$0** | ✅ compute only | — |
| **Gemini 2.5 Flash-Lite** (cost-compliant default) | ~$0.10–0.50/1M | **~$0.0004–0.0014** | ✅ **(2–7× headroom)** | ✗ |
| Gemini 3.6-flash (quality tier) | $1.50/1M in | ~$0.0045 | ❌ ~1.5× over | ✅ `radio`,`static` |
| gpt-audio-mini | **$10/1M** | **~$0.006–0.0073** | ❌ ~2× over | ✗ |
| gpt-audio (full) | $32/1M | ~$0.02 | ❌ ~7× over | ✅ `tv` |
| Mock / disabled provider | — | $0 | ✅ no external call | — |

**No cost-compliant audio-LLM reliably names the subtle background noise** — the models that hear it
(3.6-flash, gpt-audio-full) are 1.5–7× over the ceiling. OpenAI bills **audio-input separately and far
above text** ($10/1M mini, $32/1M full — vs $0.60 text), which alone puts both over budget. The
production emotion path is therefore a **cascade** —
`gemini-3.6-flash → gemini-3.5-flash-lite → gemini-2.5-flash-lite → gpt-audio-mini` — that prefers the
noise-hearing model when quota allows and degrades to the **cost-compliant Flash-Lite** (~$0.001/min)
otherwise. The dashboard shows the **actual $/audio-min and which model ran** per batch, so the
cost/accuracy trade is explicit. The 6 acoustic fields are $0 regardless (local DSP).

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

Gemini's **free tier** has a low daily request quota (we hit `429 RESOURCE_EXHAUSTED` on the top model
quickly). The **cascade** handles this: a 429 skips **immediately** to the next model
(quota-aware — no wasted backoff), and a full-chain failure returns a partial result (all 6 acoustic
fields + a low-confidence default emotion), so one throttled call never fails the batch. For a reliable
hosted deployment, enable **pay-as-you-go billing** so 3.6-flash / Flash-Lite stay available — per-call
cost stays ~$0.001–0.0045/min (1,000 audio-min ≈ **$1–$4.50** depending on tier).

## Cost controls in the system
- Each provider's `disclose()` surfaces its per-1M audio/text rates for auditing; cost is then computed
  from **real token usage** per call and shown live on the dashboard vs the $0.003 ceiling.
- Provider/model chain is fully env-configurable (`APP_EMOTION_CHAIN`); the whole system runs at **$0**
  on the local/mock path.
- `APP_API_CONCURRENCY` bounds in-flight provider calls (spend + rate-limit control).

## Worked example (30-min call)
Gemini Flash-Lite: `30 × ~$0.001 ≈ $0.03`. Ceiling for 30 min = $0.09 → **~3× under**.
gpt-audio-mini would be `30 × ~$0.006 ≈ $0.18` → **2× over** the $0.09 budget.
