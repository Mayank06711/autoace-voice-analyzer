# Validation

**Metric:** per-field agreement vs the 3 labeled calls in `data/labels.csv`. With **n=3 this is
calibration/sanity, NOT an accuracy claim** — the real evaluation is the hidden set. Unit tests use
synthetic signals with known ground truth (37 tests, all passing).

## Acoustic fields — agreement vs labels (predictions in `predictions.json`)

| Field | call_001 | call_002 | call_003 |
|---|---|---|---|
| long_silence_present | ✅ | ✅ | ✅ |
| background_noise_present | ✅ | ✅ | ✅ |
| background_noise_severity | ✅ | ✅ | ✅ |
| background_noise_type* | ✅ (none) | ~ TV vs "background chatter" | ~ static vs "background chatter" |
| audio_quality | ✅ | ✅ | ✅ |
| speaker_overlap_present | ✅ | ❌ (→ LLM) | ❌ (→ LLM) |

Acoustic agreement (of the 5 enum/bool fields): **call_001 5/5 · call_002 4/5 · call_003 4/5**.
`background_noise_type` is open text (partial credit); *the heuristic labels are approximate.

### Confusion matrix — `background_noise_present` (the key fix)
Before (VAD-gap SNR): predicted absent for all → 1 correct. After (**WADA-SNR on speech**):

|  | pred present | pred absent |
|---|---|---|
| **actual present** (002,003) | 2 | 0 |
| **actual absent** (001) | 0 | 1 |

3/3 correct. `background_noise_severity` also 3/3 (none, medium, medium).

## Measured agreement vs the 3 labels (real Gemini, `scripts/accuracy.py`)
(n=3 → calibration only, not an accuracy claim. `emotional_tone` is enum-constrained via Gemini
structured output, so it can only emit an allowed value.)

| Metric | Result |
|---|---|
| All 7 comparable fields (7×3) | **15/21 = 71%** |
| 6 acoustic (non-emotion) | **14/15 = 93%** (only miss: call_002 overlap) |
| emotional_tone exact | **0/3**; **within-1 escalation step: 2/3**; avg distance 1.33 |
| emotional_intensity | 1/3 exact; **within-1: 3/3** |

**Read:** the deterministic acoustic half is strong (93%); **emotional_tone is the weak spot** — this
run produced "frustrated" for all three (labels: upset / neutral / satisfied), i.e. 2 of 3 are only
one escalation step off but call_003 (satisfied→frustrated) is a valence miss. Likely causes: the
audio is **dual-mono (agent+customer mixed)** so the LLM can't isolate the customer, plus **n=3
subjective labels** and run-to-run LLM variance. Emotion accuracy must be judged on the hidden set
and/or by ear, NOT tuned to 3 clips. Predictions in `predictions.json` (regenerated with Gemini).

## Known misses
- `speaker_overlap_present` on 002/003 — no acoustic feature separated overlap across the calls
  (flatness/bandwidth/centroid nearly identical); handled by the LLM (`overlap_method=llm`) in the
  full path, or pyannote as an upgrade.

## Leakage / rigor
No training performed → no train/test leakage. Thresholds are documented and env-configurable so the
calibration is transparent and re-runnable. Validation across the same-call/same-speaker is moot at
n=3; leave-one-call-out is noted as the method for a larger labeled set.
