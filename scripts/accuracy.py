"""Run the real product pipeline (hybrid, Gemini) on the 3 labeled calls, save predictions.json,
and measure agreement vs data/labels.csv. Key from env.
Run: GEMINI_API_KEY=... python scripts/accuracy.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.emotion.registry import get_provider
from app.analysis.pipeline import run_file
from app.config import get_settings

TONE_ORD = {"satisfied": -1, "neutral": 0, "frustrated": 1, "upset": 2, "distressed": 3}
INT_ORD = {"low": 0, "medium": 1, "high": 2}
CALLS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
# exact-comparable fields (noise_type is open text; confidence is our own metric)
CMP = ["emotional_tone", "emotional_intensity", "background_noise_present",
       "background_noise_severity", "audio_quality", "speaker_overlap_present", "long_silence_present"]
ACOUSTIC = ["background_noise_present", "background_noise_severity", "audio_quality",
            "speaker_overlap_present", "long_silence_present"]


async def main() -> int:
    s = get_settings()
    provider = get_provider("gemini", s)
    labels = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(open("data/labels.csv"))}

    preds, exact, total, ac_exact, ac_total = {}, 0, 0, 0, 0
    tone_d, int_d = [], []
    for n in CALLS:
        r = (await run_file(Path("data") / n, provider, s))[0].model_dump(mode="json")
        preds[n] = r
        lab = labels[n]
        marks = []
        for f in CMP:
            ok = r[f] == lab[f]
            total += 1
            exact += ok
            if f in ACOUSTIC:
                ac_total += 1
                ac_exact += ok
            marks.append(f"{f}:{'OK' if ok else f'{r[f]}!={lab[f]}'}")
        tone_d.append(abs(TONE_ORD.get(r["emotional_tone"], 0) - TONE_ORD.get(lab["emotional_tone"], 0)))
        int_d.append(abs(INT_ORD.get(r["emotional_intensity"], 0) - INT_ORD.get(lab["emotional_intensity"], 0)))
        print(f"== {n} ==  " + "  ".join(marks))

    json.dump(preds, open("predictions.json", "w"), indent=2)
    print("\n--- AGREEMENT vs 3 labels (calibration, n=3) ---")
    print(f"All 7 comparable fields : {exact}/{total} = {100*exact/total:.0f}%")
    print(f"6 acoustic (non-emotion): {ac_exact}/{ac_total} = {100*ac_exact/ac_total:.0f}%")
    print(f"emotional_tone exact    : {sum(d==0 for d in tone_d)}/3 ; within-1-step: {sum(d<=1 for d in tone_d)}/3 ; avg dist {sum(tone_d)/3:.2f}")
    print(f"emotional_intensity     : {sum(d==0 for d in int_d)}/3 exact ; within-1: {sum(d<=1 for d in int_d)}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
