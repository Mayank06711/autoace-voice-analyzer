# AutoAce — Voice Tone & Background Noise Analyzer

Analyzes call audio and returns a fixed **9-field JSON** (emotional tone/intensity, background-noise
present/type/severity, audio quality, speaker overlap, long silence, confidence). Runs as a
**login-gated web dashboard**: upload a ZIP of audio + optional `labels.csv`, watch progress,
download CSV/JSON. Stays **≤ $0.003/audio-minute**, reproducible, hosted.

## Architecture (one process)
Single FastAPI/uvicorn process serves UI + API. Two field families:
- **6 acoustic fields** — deterministic local DSP (energy/VAD/spectrum → WADA-SNR, clipping, etc.). ~free, private.
- **3 emotional fields** — a **pluggable `EmotionProvider`** (`mock` | `gemini` | `openai`), overlap folded into the same LLM call.

Concurrency: asyncio + `ThreadPoolExecutor` (CPU work) + a semaphore (API calls). Models load once at
startup. SQLite (WAL) holds state; audio blobs go through a pluggable `StorageBackend` (local | s3 |
r2 | gcs | azure | supabase).

## Setup & run (Windows)
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # ffmpeg must be on PATH
.venv/Scripts/pytest                                   # 37 tests
.venv/Scripts/python scripts/predict_cli.py data/      # 9-field JSON for the 3 calls
APP_ADMIN_KEY=demo .venv/Scripts/python -m uvicorn app.main:app --port 8000
# open http://localhost:8000  → sign in with "demo" → upload a ZIP → download results
```

## Docker
```bash
docker build -t autoace .
docker run -p 8000:8000 -e APP_ADMIN_KEY=demo -e APP_EMOTION_PROVIDER=gemini -e GEMINI_API_KEY=... autoace
```

## Configuration (env; see `.env.example`)
`APP_ADMIN_KEY`, `APP_SESSION_SECRET`, `APP_EMOTION_PROVIDER` (mock|gemini|openai), `APP_MAX_WORKERS`,
`APP_STORAGE_BACKEND`, `APP_VAD_BACKEND`, plus detector thresholds — all overridable, no code change.

## Batch format
ZIP (or folder) with audio files (`ogg/wav/mp3/flac/m4a/opus`) at the root + optional `labels.csv`
(`name`, `result_json` columns). Missing manifest → unlabeled batch (predictions still produced).
One malformed file is reported and skipped; the batch continues.

## External API disclosure
Default provider is **`mock`** (no external calls, no key). If `gemini`/`openai` is selected, audio is
sent to that provider — disclosed per provider via `disclose()`. Cost-compliant cloud choice is
**Gemini 2.5 Flash-Lite** (~$0.0006/audio-min). See [COST.md](COST.md).

## Deliverables
[Technical memo](MEMO.md) · [Validation & confusion matrix](VALIDATION.md) · [Cost analysis](COST.md) ·
[Latency analysis](LATENCY.md) · [Predictions for the 3 calls](predictions.json)
