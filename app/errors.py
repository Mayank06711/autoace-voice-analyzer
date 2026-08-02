"""Typed error hierarchy. Workers convert any of these into a per-file 'failed' + reason."""


class AppError(Exception):
    """Base for all application errors."""


class DecodeError(AppError):
    """Audio could not be decoded (corrupt, unsupported, empty, too short)."""


class ProviderError(AppError):
    """An emotion provider call failed after retries."""


class ValidationError(AppError):
    """A result failed schema validation and could not be repaired."""


class IngestionError(AppError):
    """A batch could not be ingested (bad zip, no audio, malformed manifest)."""
