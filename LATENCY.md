# Latency Analysis

Two components run **concurrently** per clip: local DSP (Track A, CPU) and the emotion API (Track B,
network). Per-clip wall time ≈ **max(DSP, API)**, not the sum.

## Local DSP — measured on the 3 provided calls
(Windows, CPU-only, `mock` emotion so only local DSP is timed. Command: `run_file` per clip.)

| Clip | Audio duration | Processing time | Proc / audio-min |
|---|---|---|---|
| call_001.ogg | 30.9 s | 3.90 s* | 7.56 s* |
| call_002.ogg | 35.0 s | 0.26 s | 0.45 s |
| call_003.ogg | 171.9 s | 1.01 s | 0.35 s |
| **total** | 237.8 s | 5.17 s | **1.31 s** |

\* call_001 includes **one-time warmup / numba JIT** on first use. **Steady-state ≈ 0.35–0.45 s of CPU
per audio-minute** — warmup is paid once at server startup (`lifespan.warmup()`), not per request. A
30-min call ≈ ~12 s CPU; real-time factor ≈ **0.007×**, well within "reasonable for batch analysis."

## Emotion API — the cascade
- One network round-trip per clip, **typically ~1–3 s regardless of clip length** (audio is sent once).
- Because Track A ∥ Track B, this usually *hides under* the DSP time for long clips and dominates for
  short ones.
- **Cascade cost:** on a 429/quota the pipeline **skips to the next model immediately** (quota-aware —
  no retry/backoff on quota), so a fallback adds only ~one extra fast round-trip per skipped model
  (a few hundred ms), not a multi-second backoff. A full-chain failure still returns the 6 acoustic
  fields with a low-confidence default — a file is never lost.

## Batch throughput
Up to `APP_MAX_WORKERS` (default **3**) clips process concurrently; API calls additionally bounded by
`APP_API_CONCURRENCY` (default **10**). Rolling dispatch: the next file starts the instant one finishes.
Estimate for a hidden batch: local-only path ≈ `~0.4 s CPU × total-audio-min ÷ workers`; the hybrid
path is dominated by API round-trips, bounded by the concurrency cap. Both are comfortably within
batch-processing latency.

## Deployed environment
Runs as a **single uvicorn process** in a Docker container on Railway (Python 3.11, CPU-only,
torch-free). One process by design — models load once, one SQLite writer, internal thread pool for
parallelism; scale via `APP_MAX_WORKERS`, never extra uvicorn workers. Cold start is fast (webrtc VAD
default → no model download); Railway's healthcheck (`/`, 120 s budget) gates traffic until warm.

## Scaling
Single process is right-sized for the trial; the documented scale-out path (ProcessPool/Celery,
Postgres, object storage, Redis queue) lifts the per-machine ceiling with **no redesign** — the layers
are isolated behind interfaces (see PLAN §18).
