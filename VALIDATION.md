# Validation

**Metric:** per-field agreement vs the 3 labeled calls in `data/labels.csv`. With **n=3 this is
calibration/sanity, NOT an accuracy claim** — real evaluation is the hidden set. **52 automated tests**
pass locally (audio-decode tests + pure-logic); CI runs the **25 pure-logic tests inside the deployed
image** (the confidential samples aren't committed, so audio tests auto-skip there).

## Deterministic acoustic fields — agreement vs labels

| Field | call_001 | call_002 | call_003 |
|---|---|---|---|
| long_silence_present | ✅ | ✅ | ✅ |
| background_noise_present | ✅ | ✅ | ✅ |
| background_noise_severity | ✅ | ✅ | ✅ |
| audio_quality | ✅ | ✅ | ✅ |
| speaker_overlap_present | ✅ | ❌ (→ LLM) | ✅ |
| background_noise_type* | ✅ (none) | ~ TV vs `ambient noise`(DSP) / `radio`(3.6-flash) | ~ static vs `static/hiss`(DSP) / `static`(3.6-flash) |

Agreement on the 5 enum/bool acoustic fields: **14/15 = 93%** (only miss: call_002 overlap by the
heuristic — deferred to the LLM in the full path). `background_noise_type` is open text, scored
loosely; *the DSP labels are approximate, and the top cascade model (`gemini-3.6-flash`) names them
correctly (`radio`,`static`) when quota allows.

### Confusion matrix — `background_noise_present` (the key fix)
Before (VAD-gap SNR): predicted absent for all → 1 correct. After (**WADA-SNR on speech-active
samples**):

|  | pred present | pred absent |
|---|---|---|
| **actual present** (002, 003) | 2 | 0 |
| **actual absent** (001) | 0 | 1 |

3/3 correct. `background_noise_severity` also 3/3 (none, medium, medium).

## Emotional tone — the hard, subjective field
`emotional_tone` is enum-constrained via structured output, so it can only emit an allowed value. But
**n=3 cannot establish emotion accuracy**, and it's genuinely hard: realistic SER on real call audio
tops out ~0.35–0.45 macro-F1, human inter-annotator agreement is only 60–80%, and our audio is
**dual-mono** (agent + customer mixed), so the model can't cleanly isolate the customer. Across runs
both the LLM and wav2vec2 **over-read negativity** on the two subtle calls where the *words* are neutral
(a calm appointment booking; a plain "Spanish please"). Emotion must be judged on the hidden set and/or
by ear — **not tuned to 3 clips**. This is why we ship a fairer scorer, not a fitted threshold:

## In-product comparison (predictions vs labels)
When a batch's `labels.csv` has ground truth, the dashboard's **"Compare vs labels"** shows two views:
- **Exact-match** per field (deterministic, strict).
- **LLM-semantic** (a `gpt-4o-mini` text judge) that treats near-equivalents as agreement —
  `TV`≈`radio`, `static`≈`hiss`, `frustrated`≈`annoyed`, adjacent intensities as partial.

The gap between them is itself the signal: e.g. on the samples `background_noise_type` scores **0%
exact but ~50% semantic** (right gist, wrong wording), which is the honest way to score a subjective,
free-text field instead of penalizing a correct-in-spirit answer as fully wrong.

## Leakage / rigor
No training performed → no train/test leakage. Thresholds are documented and env-configurable so the
calibration is transparent and re-runnable. At n=3, same-call/same-speaker splitting is moot;
**leave-one-call-out** is the documented method for a larger labeled set. Predictions for the 3 calls
are in `predictions.json`.
