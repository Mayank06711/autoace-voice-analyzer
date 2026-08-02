# Multi-stage, CPU-only, torch-free by default → small image.

# ---- builder: install pinned wheels into a venv ----
FROM python:3.11-slim AS builder
WORKDIR /build
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- final: slim runtime + ffmpeg + app ----
FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 APP_PORT=8000
WORKDIR /app
COPY app ./app
COPY scripts ./scripts
COPY models ./models
RUN mkdir -p storage logs   # sqlite db + per-run logs live here at runtime
EXPOSE 8000
# Bind to $PORT when the host injects one (Railway/Render/Cloud Run assign it dynamically),
# else default to 8000 for local `docker run`. Shell form so the var expands.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/').read()" || exit 1
# ONE worker by design (models load once; single SQLite writer). Never --workers N.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
