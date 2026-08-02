"""Single source of configuration. Loaded once from env/.env; nothing else reads os.environ."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Prefix APP_ (except provider API keys)."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    # --- env / auth ---
    env: str = "dev"                   # dev | production (production refuses default secrets)
    admin_key: str = "changeme"
    session_secret: str = "dev-secret-change-me"
    max_upload_mb: int = 200           # reject uploads larger than this

    # --- emotion (vendor-neutral) ---
    emotion_mode: str = "api"          # api | local | disabled
    emotion_provider: str = "mock"     # mock | openai | gemini | ...  (primary; used if no chain)
    emotion_fallback: str = ""         # secondary provider used if the primary fails (e.g. openai)
    # Ordered "provider:model" cascade tried best→cheapest; overrides provider/fallback when set.
    # e.g. gemini:gemini-3.6-flash,gemini:gemini-3.5-flash-lite,gemini:gemini-2.5-flash-lite,openai:gpt-audio-mini
    emotion_chain: str = ""
    emotion_model: str = ""            # generic override; each provider takes it only if it matches
    openai_model: str = ""             # per-provider model override (so a chain can set both)
    gemini_model: str = ""             # per-provider model override
    emotion_two_pass: bool = False     # isolate+describe the customer, then classify (2 API calls)
    compare_model: str = "gpt-4o-mini"  # cheap TEXT model that judges predictions vs labels (not gemini)

    # --- detectors / overlap ---
    overlap_method: str = "auto"       # auto | llm | heuristic | pyannote
    noise_type_enabled: bool = True
    vad_backend: str = "webrtc"        # webrtc (definite) | silero (upgrade)

    # --- server / concurrency ---
    port: int = 8000
    max_workers: int = 3
    api_concurrency: int = 10
    file_timeout_s: int = 120

    # --- audio / dsp params (all calibratable via env) ---
    sample_rate: int = 16000
    frame_ms: int = 25
    hop_ms: int = 10
    energy_floor_dbfs: float = -45.0
    long_silence_s: float = 5.0
    vad_threshold: float = 0.5

    # noise (WADA-SNR on speech samples; calibrated on the 3 labeled calls, verify on hidden set)
    noise_wada_present_db: float = 88.0   # >= this = no meaningful noise
    noise_wada_low_db: float = 80.0       # [low, present) = low severity
    noise_wada_med_db: float = 50.0       # [med, low) = medium; < med = high

    # audio_quality (independent of noise per the brief: clipping / muffling / volume only)
    clip_warn: float = 0.005
    clip_bad: float = 0.02
    muffle_warn_hz: float = 1500.0
    muffle_bad_hz: float = 900.0
    low_volume_dbfs: float = -55.0

    # --- retry / backoff (provider calls) ---
    retry_max: int = 4
    retry_base_s: float = 1.0
    retry_cap_s: float = 15.0
    api_timeout_s: float = 30.0

    # --- storage (pluggable backend) / db / logs ---
    db_url: str = "sqlite:///storage/app.db"
    storage_backend: str = "local"     # local | s3 | r2 | gcs | azure | supabase
    storage_dir: str = "storage"       # used by local backend
    storage_bucket: str = ""           # used by cloud backends
    storage_prefix: str = "batches"    # key prefix within the bucket/dir
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_json: bool = True

    # --- provider credentials (no APP_ prefix) ---
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
    gemini_api_key: str = Field("", validation_alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — the only configuration source in the app."""
    return Settings()
