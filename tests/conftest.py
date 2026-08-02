"""Pytest bootstrap.

The 3 sample calls in data/ are confidential and NOT committed, so on a clean checkout (e.g. public
CI) the tests that decode real audio can't run. We skip exactly those files when the samples are
absent — the whole pure-logic suite (schema, config, providers, merge, pricing, fallback chain,
comparison, repositories, ...) still runs. With data/ present (local dev) everything runs.
"""
from pathlib import Path

_HAS_SAMPLES = (Path(__file__).resolve().parent.parent / "data" / "call_001.ogg").exists()

if not _HAS_SAMPLES:
    collect_ignore = [
        "unit/test_io.py",
        "unit/test_features.py",
        "unit/test_detectors.py",
        "integration/test_pipeline.py",
        "integration/test_batch.py",
        "integration/test_api.py",
    ]
