"""Experiment: open-source wav2vec2 SER (the brief's 'acoustic model vs foundation model' comparison).

Runs superb/wav2vec2-base-superb-er (IEMOCAP 4-class) locally, maps to our taxonomy, compares to the
labels and to Gemini's read. Needs torch + transformers (temporary). Run: python scripts/wav2vec_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from transformers import pipeline

from app.audio.io import decode
from app.config import get_settings

# IEMOCAP classes -> our 5-way taxonomy (best-effort; there is no exact 'frustrated'/'satisfied')
MAP = {"neu": "neutral", "hap": "satisfied", "ang": "upset", "sad": "distressed"}
CSV_LABEL = {"call_001.ogg": "upset", "call_002.ogg": "neutral", "call_003.ogg": "satisfied"}
YOUR_EAR = {"call_003.ogg": "neutral"}  # user listened: 003 is neutral (bot barge-in issue)
GEMINI = {"call_001.ogg": "upset/neutral*", "call_002.ogg": "frustrated", "call_003.ogg": "frustrated"}


def main() -> int:
    s = get_settings()
    clf = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    for n in ["call_001.ogg", "call_002.ogg", "call_003.ogg"]:
        c = decode(Path("data") / n, s)
        out = clf({"raw": c.samples.astype(np.float32), "sampling_rate": c.sr}, top_k=4)
        ranked = [(o["label"], round(o["score"], 2)) for o in out]
        mapped = MAP.get(out[0]["label"], out[0]["label"])
        print(f"{n}")
        print(f"   csv_label={CSV_LABEL[n]}  your_ear={YOUR_EAR.get(n, CSV_LABEL[n])}  gemini={GEMINI[n]}")
        print(f"   wav2vec2 raw={ranked}  -> mapped: {mapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
