"""CLI: run the FULL 9-field analysis over a folder or single file and print JSON.

Uses the configured emotion provider (mock by default → runs with no API key).
Usage: python scripts/predict_cli.py data/
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow direct invocation

from app.analysis.emotion.registry import get_provider
from app.analysis.pipeline import run_file
from app.audio.io import is_supported
from app.config import get_settings
from app.errors import DecodeError


async def _run(files: list[Path], s) -> dict[str, dict]:
    provider = get_provider(s=s)
    out: dict[str, dict] = {}
    for f in files:
        try:
            result, _ = await run_file(f, provider, s)
            out[f.name] = result.model_dump(mode="json")
        except DecodeError as e:
            out[f.name] = {"error": str(e)}
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/predict_cli.py <folder-or-file>")
        return 1
    s = get_settings()
    target = Path(argv[1])
    files = [target] if target.is_file() else sorted(
        p for p in target.iterdir() if p.is_file() and is_supported(p.name)
    )
    print(json.dumps(asyncio.run(_run(files, s)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
