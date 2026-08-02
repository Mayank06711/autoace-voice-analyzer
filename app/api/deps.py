"""Auth guard dependency — attached to every protected route (incl. download)."""
from __future__ import annotations

from fastapi import HTTPException, Request

from app.api import auth
from app.config import get_settings


def require_session(request: Request) -> bool:
    s = get_settings()
    if not auth.valid_session(request.cookies.get(auth.COOKIE), s):
        raise HTTPException(status_code=401, detail="unauthorized")
    return True
