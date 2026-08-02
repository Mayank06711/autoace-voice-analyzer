# Technical Memo

## Problem
Per call clip, emit a fixed 9-field JSON (emotion tone/intensity + background-noise present/type/
severity + audio quality + speaker overlap + long silence + confidence), accurate on a hidden set,
≤ $0.003/audio-min, reproducible, delivered as a hosted, login-gated dashboard.

## Approaches considered
1. **End-to-end multimodal LLM** — one call returns all 9 fields. Simple, but costlier, less
   consistent/explainable on the physical fields, and it hallucinated deterministic fields (see below).
2. **Fully-local (DSP + open SER/audio-tagging models)** — $0 and fully private, but off-the-shelf SER
   can't emit the 5-class taxonomy and audio-tagging (YAMNet) missed the real noises (see experiments).
3. **Hybrid (chosen)** — deterministic DSP for the 6 acoustic fields + a **provider cascade** for the 3
   emotional fields. Best accuracy-per-cost, each field judged by the right tool, degrades gracefully
   (cheaper model → local mock) when a key or quota is exhausted.

## Final architecture
- **Acoustic (local, deterministic, $0):** one shared feature front-end (energy dBFS, VAD, STFT
  spectrum) feeds six independent detectors.
  - `long_silence`: longest non-speech + low-energy run.
  - `background_noise_present/severity`: **WADA-SNR on speech-active samples** (detects noise *under*
    speech, which VAD-gap methods miss), gated on "has speech".
  - `audio_quality`: clipping + muffling + low-volume scorecard — **independent of noise** (per brief).
  - `background_noise_type`: spectral heuristic (static/hiss · hum/mechanical · hi-freq hiss · ambient).
  - `speaker_overlap`: heuristic fallback; real detection comes from the LLM (mono overlap is neural).
- **Emotion (provider cascade, quota-aware fallback):** vendor-neutral `EmotionProvider` + registry.
  The default chain is tried best→cheapest→cross-provider:
  **`gemini-3.6-flash → gemini-3.5-flash-lite → gemini-2.5-flash-lite → gpt-audio-mini`**.
  A 429/quota error skips straight to the next model (no wasted backoff); a full-chain failure yields a
  safe partial (6 acoustic fields + low-confidence default). The prompt is structured with the brief's
  own definitions, customer-focused, with an acoustic-forcing chain-of-thought and a "most calls are
  neutral" calibration guard. The provider also returns overlap + a noise-type description.
- **`merge.py` is the arbiter:** picks the overlap source (LLM vs heuristic), uses the LLM's
  `noise_type` **only when DSP confirms noise AND the value is a real descriptor** (a sentinel guard
  rejects `"none"`/`""` so a "no-noise" answer never becomes a type), blends confidence, and repairs to
  the schema so a malformed provider reply never crashes a file.
- **Cost is measured, not estimated:** every provider call records its real token counts
  (`audio_tokens`, text in/out) × the vendor's per-token rate; the dashboard shows **$/audio-minute vs
  the $0.003 ceiling per batch**, and flags when a model fell back.
- **Validation is in-product:** when a batch's `labels.csv` carries ground truth, a "Compare vs labels"
  view shows **exact-match AND an LLM-judged semantic score** (so `TV`≈`radio`, `static`≈`hiss`,
  `frustrated`≈`annoyed` count as agreement — fairer than exact match on subjective/free-text fields).
- **Runtime:** single uvicorn process; asyncio + thread pool; models load once; SQLite (WAL);
  pluggable storage. Dockerized; deployed on Railway; GitHub CI builds the image and tests inside it.

## Experiments (the §9 "≥2 materially different approaches" comparison)

**A. wav2vec2 SER vs the audio-LLM (emotion).** Ran `superb/wav2vec2-base-superb-er` (IEMOCAP 4-class)
locally: it predicted **"angry" for all three** calls (strong negativity bias), **cannot emit the
required taxonomy** (no `frustrated`/`satisfied` class), needs **torch + ~360 MB**, and is
domain-mismatched (acted studio speech vs real noisy calls). → An **instructable 5-class audio-LLM +
deterministic DSP** beats a fixed SER model here.

**B. YAMNet (audio-event tagging) vs DSP, for `background_noise_type`.** Ran reference YAMNet (521
AudioSet classes) on the isolated non-speech of each call. On the two noisy calls it labeled the
background **~90–98% "Silence"** — the TV/static sit *under* the speech and are too faint for even a
purpose-built tagger to name. → YAMNet does **not** help on this data; the coarse DSP heuristic is no
worse and is free.

**C. Which model actually hears the noise (and what it costs).** Measured from real API token usage:

| Model | hears TV/static? | $/audio-min (measured) | ≤ $0.003? |
|---|---|---|---|
| gpt-audio (full) | ✓ `tv` | ~$0.02 (audio-in $32/1M) | ❌ ~7× |
| **gemini-3.6-flash** | ✓ `radio`,`static` | ~$0.0045 (input $1.50/1M) | ❌ ~1.5× |
| gpt-audio-mini | ✗ `none` | ~$0.006 (audio-in **$10/1M**, not the $0.60 text rate) | ❌ ~2× |
| **gemini-2.5-flash-lite** | ✗ `""` | **~$0.0004–0.0014** | ✅ |
| YAMNet / DSP (local) | ✗ (silence) | **$0** | ✅ |

**The key finding: no cost-compliant model reliably names the subtle noise** — the models that hear it
are 1.5–7× over the ceiling. This is exactly why the emotion path is a **cascade**: it tries the
noise-hearing `gemini-3.6-flash` first (great when quota allows), and degrades to the cost-compliant
`flash-lite` (with the DSP noise label) otherwise. The dashboard shows, per batch, which model ran and
the real $/min — so the cost/accuracy trade is explicit, not hidden.

## Confidence model
Overall `confidence` is a defined function, not the LLM's raw number (which only covers emotion and is
over-confident): `confidence = 0.6·emotion_conf + 0.4·acoustic_certainty`, where
`acoustic_certainty = quality_factor(clear 1.0 / slightly 0.7 / severe 0.4) · (0.5 + 0.5·min(1, speech_frac/0.3))`.
It reaches 1.0 only when the LLM is sure AND audio is clear AND there's ample speech; it falls toward
~0.5 when emotion is ambiguous or the audio is impaired/low-speech. Every value is traceable.

## Cost, privacy & API disclosure (per §11)
- **Paid APIs used:** Google Gemini (`gemini-3.6-flash`, `-3.5/2.5-flash-lite`) and OpenAI
  (`gpt-audio-mini`; `gpt-4o-mini` only for the offline label-comparison, never for audio).
- **Cost model:** per-token, from vendor pricing pages, computed from real usage — see COST.md.
  Cost-compliant production default is **Gemini Flash-Lite ~$0.001/min**; the acoustic half is $0.
- **Does audio leave AutoAce infra?** **Yes**, in the API path — audio is sent to Google/OpenAI. The
  brief forbids uploading to *unapproved* services, so those vendors must be approved, **or** run the
  built-in **local path** (`APP_EMOTION_MODE=disabled`) which keeps audio in-infra, emits the 6
  acoustic fields deterministically, and costs $0. Retention follows each vendor's API policy
  (zero-retention available under their enterprise terms).

## Failure modes & limitations
- **`background_noise_type` is intrinsically hard** — subtle in-call noise is missed by cheap models,
  by YAMNet, and by DSP alike; only over-ceiling models name it. Reported honestly (coarse DSP label
  or the top-model description when the cascade reaches it).
- **`emotional_tone` is the weak, subjective field** — realistic SER on real calls tops out ~0.35–0.45
  macro-F1; both the LLM and wav2vec2 over-read negativity when the *words* are neutral. A transcript/
  content signal is the most promising fix.
- **Audio egress** in the API path (mitigated by the local mode above).
- **n=3 calibration** — thresholds are set on 3 clips; real accuracy is the hidden set only.
- **Gemini free-tier quota** — the top model (3.6-flash) has limited free quota; the cascade + billing
  handle this, but a demo on the free tier may fall back to cheaper models.

## Next steps
Add a transcript/content channel to correct prosody over-read on emotion (biggest expected gain);
enable Gemini billing so the noise-hearing 3.6-flash is the steady primary within a per-call budget;
Silero-ONNX VAD (higher accuracy than webrtc); pyannote overlap upgrade; calibrate thresholds and
confidence on a larger labeled set with leave-one-call-out.
