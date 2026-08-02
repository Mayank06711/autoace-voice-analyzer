# Latency Analysis

Measured on the 3 provided calls (Windows, Python 3.12, CPU-only, `mock` emotion so only local DSP is
timed). Command: `run_file` per clip.

| Clip | Audio duration | Processing time | Proc / audio-min |
|---|---|---|---|
| call_001.ogg | 30.9 s | 3.90 s* | 7.56 s* |
| call_002.ogg | 35.0 s | 0.26 s | 0.45 s |
| call_003.ogg | 171.9 s | 1.01 s | 0.35 s |
| **total** | 237.8 s | 5.17 s | **1.31 s** |

\* call_001 includes **one-time warmup / numba JIT** on first use. **Steady-state ≈ 0.35–0.45 s of
CPU per audio-minute** — the warmup cost is paid once at server startup (`lifespan.warmup()`), not per
request.

## Interpretation
- **Local DSP** is fast and roughly constant per audio-minute (~0.4 s/min steady-state) → a 30-min
  call ≈ ~12 s CPU. Real-time factor ≈ 0.007× (well within "reasonable for batch analysis").
- **Emotion API** (when enabled) adds one network round-trip per clip — typically ~1–3 s regardless of
  clip length. Because Track A (DSP) and Track B (API) run concurrently, per-clip wall time ≈
  max(DSP, API), not the sum.
- **Batch throughput:** up to `APP_MAX_WORKERS` (default 3) clips process concurrently; API calls
  additionally bounded by `APP_API_CONCURRENCY` (default 10).

## Estimate for a hidden batch
Local-only path: `~0.4 s CPU × total-audio-minutes ÷ workers`. Hybrid path is dominated by the API
round-trips, bounded by the concurrency cap. Both are comfortably within batch-processing latency.

## Scaling
Single process is right-sized for the trial; the documented scale-out path (ProcessPool/Celery,
Postgres, object storage) lifts the per-machine ceiling with no redesign (see PLAN §18).
