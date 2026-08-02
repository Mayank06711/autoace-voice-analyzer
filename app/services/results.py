"""Export batch results as CSV or JSON, preserving filenames and the manifest shape."""
from __future__ import annotations

import csv
import io
import json

from app.infra import repositories as repo


def export(batch_id: str, fmt: str = "json") -> tuple[str, str, bytes]:
    """Return (media_type, filename, body_bytes)."""
    files = repo.list_files(batch_id)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["name", "result_json"])
        for f in files:
            w.writerow([f["name"], f["result_json"] or ""])
        return "text/csv", f"{batch_id}.csv", buf.getvalue().encode("utf-8")

    data = [
        {
            "name": f["name"],
            "result_json": json.loads(f["result_json"]) if f["result_json"] else None,
            "error": f["error"],
        }
        for f in files
    ]
    return "application/json", f"{batch_id}.json", json.dumps(data, indent=2).encode("utf-8")
