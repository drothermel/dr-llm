from __future__ import annotations


class StreamingLogError(RuntimeError):
    """Base error for streaming-log operations."""


class PayloadIntegrityError(StreamingLogError):
    """Stored payload bytes do not match the expected content hash."""


class StreamingLogResourceError(StreamingLogError):
    """A NATS resource exists but is incompatible with the expected contract."""


__all__ = [
    "PayloadIntegrityError",
    "StreamingLogError",
    "StreamingLogResourceError",
]
