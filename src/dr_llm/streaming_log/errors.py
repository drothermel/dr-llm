from __future__ import annotations


class StreamingLogError(RuntimeError):
    """Base error for streaming-log operations."""


class PayloadIntegrityError(StreamingLogError):
    """Stored payload bytes do not match the expected content hash."""


class PayloadNotFoundError(StreamingLogError):
    """A referenced payload object was not found in storage."""


class PayloadReadError(StreamingLogError):
    """A referenced payload object could not be read from storage."""


class StreamingLogResourceError(StreamingLogError):
    """A NATS resource exists but is incompatible with the expected contract."""


__all__ = [
    "PayloadIntegrityError",
    "PayloadNotFoundError",
    "PayloadReadError",
    "StreamingLogError",
    "StreamingLogResourceError",
]
