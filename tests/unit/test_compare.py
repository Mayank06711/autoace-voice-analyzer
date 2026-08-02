"""Deterministic exact-match agreement (the free baseline shown next to LLM-semantic)."""
from __future__ import annotations

import json

from app.services.compare import exact_agreement


def _pair(name, pred, label):
    return {"name": name, "predicted": json.dumps(pred), "label": json.dumps(label)}


def test_exact_match_and_the_synonym_gap():
    pairs = [_pair(
        "call_002.ogg",
        {"emotional_tone": "neutral", "background_noise_present": True,
         "background_noise_type": "radio", "speaker_overlap_present": True},
        {"emotional_tone": "neutral", "background_noise_present": True,
         "background_noise_type": "TV", "speaker_overlap_present": True},
    )]
    ex = exact_agreement(pairs)
    assert ex["emotional_tone"]["pct"] == 100          # exact hit
    assert ex["background_noise_present"]["pct"] == 100  # bool hit
    assert ex["speaker_overlap_present"]["pct"] == 100
    # 'radio' vs 'TV' is 0% by EXACT match — this is exactly the case the LLM-semantic view rescues
    assert ex["background_noise_type"]["pct"] == 0


def test_only_labeled_fields_are_counted():
    # label omits most fields → only the provided ones are scored (no false penalties)
    pairs = [_pair("x.ogg", {"emotional_tone": "upset", "audio_quality": "clear"},
                   {"emotional_tone": "upset"})]
    ex = exact_agreement(pairs)
    assert ex["emotional_tone"] == {"match": 1, "total": 1, "pct": 100}
    assert ex["audio_quality"]["total"] == 0  # not in the label → not counted
