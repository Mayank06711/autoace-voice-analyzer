"""Server-side auth: verify the static admin key, issue/verify a signed session cookie."""
from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import Settings

COOKIE = "aa_session"
MAX_AGE = 86400  # 24h


def _serializer(s: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(s.session_secret, salt="autoace-auth")


def verify_admin(key: str, s: Settings) -> bool:
    return bool(key) and key == s.admin_key


def verify_login(username: str, password: str, s: Settings) -> bool:
    """Traditional username + password: username is case-insensitive (default 'admin'); the
    password is the admin key. Both must match."""
    user_ok = bool(username) and username.strip().lower() == (s.admin_user or "admin").strip().lower()
    return user_ok and verify_admin(password, s)


def make_token(s: Settings) -> str:
    return _serializer(s).dumps({"ok": True})


def valid_session(token: str | None, s: Settings) -> bool:
    if not token:
        return False
    try:
        _serializer(s).loads(token, max_age=MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False
