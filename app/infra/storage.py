"""Pluggable storage. LocalStorage is the default; s3/r2/gcs/azure/supabase slot in behind the ABC.

The analysis pipeline never assumes local disk — it calls `localize(key)` to get a readable path
(a temp download for cloud backends).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

from app.config import Settings, get_settings


class StorageBackend(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """Persist bytes under `key`; return the key."""

    @abstractmethod
    def open(self, key: str) -> bytes:
        ...

    @abstractmethod
    def localize(self, key: str):
        """Context manager yielding a LOCAL filesystem path for `key` (temp-download for cloud)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...


class LocalStorage(StorageBackend):
    def __init__(self, s: Settings):
        self.root = Path(s.storage_dir) / s.storage_prefix
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.root / key

    def save(self, key: str, data: bytes) -> str:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def open(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    @contextmanager
    def localize(self, key: str):
        yield str(self._p(key))  # already local

    def delete(self, key: str) -> None:
        try:
            self._p(key).unlink()
        except FileNotFoundError:
            pass

    def exists(self, key: str) -> bool:
        return self._p(key).exists()


# name -> backend class. Cloud backends register here (each imports its own SDK lazily).
_REGISTRY: dict[str, type[StorageBackend]] = {"local": LocalStorage}


def get_storage(s: Settings | None = None) -> StorageBackend:
    s = s or get_settings()
    cls = _REGISTRY.get(s.storage_backend, LocalStorage)
    return cls(s)
