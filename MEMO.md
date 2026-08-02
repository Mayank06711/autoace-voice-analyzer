# Technical Memo

## Problem
Per call clip, emit a fixed 9-field JSON (emotion tone/intensity + background-noise present/type/
severity + audio quality + speaker overlap + long silence + confidence), accurate on a hidden set,
≤ $0.003/audio-min, reproducible, delivered as a hosted dashboard.

## Approaches considered
1. **End-to-end multimodal LLM** — one API call returns all 9 fields. Simple, but costlier, less
   consistent/explainable on the acoustic fields, and dependent on an external API for everything.
2. **Fully-local (DSP + open models)** — $0, fully private, but the 5-class emotion taxonomy is a
   poor fit for off-the-shelf SER models (trained on acted studio emotions).
3. **Hybrid (chosen)** — deterministic DSP for the 6 acoustic fields + a pluggable LLM for the 3
   emotional fields (overlap folded into the same LLM call). Best accuracy-per-cost, each field judged
   by the right tool, and it degrades gracefully (mock/local) when no API is available.

## Final architecture
- **Acoustic (local, deterministic):** one shared feature front-end (energy dBFS, VAD, STFT spectrum)
  feeds six independent detectors.
  - `long_silence`: longest non-speech + low-energy run.
  - `background_noise_present/severity`: **WADA-SNR on speech-active samples** (detects noise *under*
    speech, which VAD-gap methods miss), gated on "has speech".
  - `audio_quality`: clipping + muffling + volume scorecard — **independent of noise** (per the brief).
  - `background_noise_type`: spectral heuristic (YAMNet-ONNX is the documented upgrade).
  - `speaker_overlap`: heuristic fallback; real detection is the LLM (mono overlap is a neural task).
- **Emotion (pluggable):** vendor-neutral `EmotionProvider` + registry (`mock`/`gemini`/`openai`).
  Structured prompt with the brief's definitions, customer-focused. Provider also returns overlap.
- **merge.py** is the arbiter (picks overlap source, gates `noise_type`, clamps/repairs to schema).
- **Runtime:** single process; asyncio + thread pool; models load once; SQLite (WAL); pluggable storage.

## Why these choices
- WADA-SNR is the standard *no-reference* SNR method — correct when there is no clean reference.
- Keeping quality independent of noise directly follows the brief's "scored separately" note.
- Overlap is left to the LLM because no acoustic feature separated it across the labeled calls
  (confirmed empirically) — forcing a threshold on n=3 would overfit.
- Vendor-neutral providers + pluggable storage make cost/privacy a config choice, not a rewrite.

## Empirical: open-source wav2vec2 SER vs audio-LLM (the §9 "two approaches" comparison)
Ran `superb/wav2vec2-base-superb-er` (IEMOCAP 4-class) locally on the 3 calls:
- call_001 → ang 0.88 → upset ✓ (matches label + human ear)
- call_002 → ang 0.54 → upset ✗ (label/ear: neutral)
- call_003 → ang 0.50 → upset ✗ (label: satisfied; human ear: neutral)

**Findings:** wav2vec2 SER (1) predicted "angry" for **all three** calls (strong negativity bias, worse
than the LLM), (2) **cannot express the required taxonomy** — no `frustrated`/`satisfied` class, so it
can never emit two of the five required tones, and (3) needs **torch + a ~360 MB model** and is
**domain-mismatched** (trained on acted studio speech; our audio is real, noisy, with agent barge-in
glitches). Conclusion: a fixed SER model is a poor fit for this 5-class taxonomy on real audio → the
**instructable audio-LLM (5-class, consistent) + deterministic DSP** hybrid is the better approach.
Note: both the LLM and wav2vec2 **over-read negativity** on the two subtle calls, while the *words*
(a calm appointment booking; a plain "Spanish please") say neutral — evidence that adding a
**transcript/content signal** would correct the prosody over-read (a concrete next improvement).

## Confidence model
Overall `confidence` is a defined function, not the LLM's raw number (which only covers emotion and
is over-confident):
`confidence = 0.6·emotion_conf + 0.4·acoustic_certainty`, where
`acoustic_certainty = quality_factor(clear=1.0 / slightly=0.7 / severe=0.4) · (0.5 + 0.5·min(1, speech_frac/0.3))`.
It reaches 1.0 only when the LLM is fully sure AND audio is clear AND there's ample speech; it falls
toward ~0.5 when either the emotion is ambiguous or the audio is impaired/low-speech. Every value is
traceable to those parameters.

## Empirical: hybrid vs all-LLM (both with structured output)
Running Gemini on all 9 fields hallucinated `long_silence_present = true` on **all three** calls and
was inconsistent on `background_noise_present`, while the deterministic DSP matched the labels on both.
Conversely the LLM produced better `background_noise_type` descriptions ("radio/TV" vs our coarse
"chatter"). Conclusion: **DSP owns the physical fields, the LLM owns emotion + noise-type description** —
the hybrid, exactly as built. **Implemented:** the LLM now supplies `background_noise_type` (returned in
the same emotion call), and `merge.py` uses it **only when DSP confirms noise is present** (else ""),
falling back to the DSP heuristic if no LLM is active. Live result: call_002 → "radio" (was "chatter").

## Failure modes & limitations
- **Emotion depends on a provider key**; without one the pipeline runs but emotion is a mock placeholder.
- **Overlap heuristic is weak** offline; accuracy needs the LLM or pyannote.
- **noise_type** is coarse without YAMNet.
- **n=3 calibration** — thresholds are calibrated on 3 clips; real validation is the hidden set.
- **Long files** are chunked to bound memory; very long calls increase latency linearly.

## Next steps
Wire YAMNet-ONNX for `noise_type`; add pyannote as an overlap upgrade; calibrate thresholds on a
larger labeled set; add Silero-ONNX VAD (higher accuracy than webrtc); confidence calibration.
