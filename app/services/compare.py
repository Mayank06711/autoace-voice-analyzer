"""Compare a batch's predictions against labels.csv ground truth (shown only when labels exist).

Two complementary views:
  1. EXACT-match per field — deterministic, free, strict (string-equal after normalising).
  2. SEMANTIC agreement — an LLM judges near-equivalents as matches ("frustrated"≈"annoyed",
     "TV"≈"radio", "static"≈"hiss"), which is fairer for subjective tone + open-text noise_type.

The gap between the two is itself informative (e.g. exact 33% but semantic 90% on noise_type means we
were RIGHT in gist, wrong in wording). The LLM judge uses OpenAI (a cheap text call) so it never spends
the Gemini quota reserved for the audio pipeline.
"""
from __future__ import annotations

import json
import logging

from app.config import Settings, get_settings
from app.infra import repositories as repo
from app.logging_conf import event

logger = logging.getLogger(__name__)

NINE = ["emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
        "background_noise_severity", "audio_quality", "speaker_overlap_present",
        "long_silence_present", "confidence"]
CATEGORICAL = [f for f in NINE if f != "confidence"]  # confidence is a self-estimate, not graded


def _norm(v) -> str:
    return str(v).strip().lower()


def exact_agreement(pairs: list[dict]) -> dict:
    """Per-field exact-match %, counting only fields the label actually provides."""
    counts = {f: [0, 0] for f in CATEGORICAL}
    for p in pairs:
        try:
            pred, lab = json.loads(p["predicted"]), json.loads(p["label"])
        except Exception:
            continue
        for f in CATEGORICAL:
            if f in lab and lab.get(f) not in (None, ""):
                counts[f][1] += 1
                counts[f][0] += int(_norm(pred.get(f)) == _norm(lab.get(f)))
    return {f: {"match": c[0], "total": c[1], "pct": (round(100 * c[0] / c[1]) if c[1] else None)}
            for f, c in counts.items()}


_FIELDS_CSV = ", ".join(f for f in NINE if f != "confidence")

_SYS = f"""You grade a call-audio analysis system. You receive several calls; each has PREDICTED \
(our 9-field JSON) and LABEL (human ground truth). Judge SEMANTIC agreement, treating synonyms and \
near-equivalents as agreement:
- emotional_tone: frustrated=annoyed=irritated; upset=angry; distressed=overwhelmed. Neutral is distinct.
- emotional_intensity: adjacent levels (low/medium) are partial agreement (~60), not 0.
- background_noise_type is FREE TEXT: reward correct GIST — tv=television=radio=background media; \
static=hiss=line noise; count "" vs a type as disagreement only if noise is actually present.
- booleans/enums (present/severity/quality/overlap/silence): agree only if they mean the same.

Return ONLY JSON. "fields" MUST be keyed by these EXACT field names (NOT by call name), each value \
aggregating agreement ACROSS ALL calls for that field:
{{"fields": {{ {_FIELDS_CSV} : each -> {{"agree_pct": 0-100, "note": "<=8 words"}} }}, \
"overall_pct": 0-100, "summary": "one sentence"}}
Example: {{"fields": {{"emotional_tone": {{"agree_pct": 67, "note": "2 of 3 match"}}, ...}}, \
"overall_pct": 74, "summary": "..."}}"""


def semantic_agreement(pairs: list[dict], s: Settings) -> tuple[dict | None, str | None]:
    """LLM-judged semantic agreement. Returns (report, model) or (None, None) if unavailable."""
    if not s.openai_api_key:
        return None, None
    from openai import OpenAI

    client = OpenAI(api_key=s.openai_api_key, timeout=s.api_timeout_s)
    body = "\n\n".join(f"CALL {p['name']}\n  PREDICTED: {p['predicted']}\n  LABEL: {p['label']}"
                       for p in pairs)
    model = s.compare_model or "gpt-4o-mini"
    try:
        r = client.chat.completions.create(
            model=model, response_format={"type": "json_object"}, temperature=0,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": "Compare these calls field-by-field:\n\n" + body}])
        return json.loads(r.choices[0].message.content or "{}"), model
    except Exception as e:  # comparison is a nice-to-have → never surface as a hard error
        event(logger, logging.WARNING, "compare.llm_failed", reason=str(e))
        return None, None


def run_comparison(batch_id: str, s: Settings | None = None) -> dict:
    """Full report: exact + semantic agreement for the labeled files in a batch."""
    s = s or get_settings()
    pairs = repo.comparison_pairs(batch_id)
    if not pairs:
        return {"n": 0, "exact": {}, "semantic": None, "semantic_model": None}
    semantic, model = semantic_agreement(pairs, s)
    return {"n": len(pairs), "exact": exact_agreement(pairs),
            "semantic": semantic, "semantic_model": model}
