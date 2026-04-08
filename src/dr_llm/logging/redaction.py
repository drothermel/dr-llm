from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_REDACT_KEYS = {
    "authorization",
    "bearer",
    "credentials",
    "api_key",
    "apikey",
    "x_api_key",
    "private_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "session_token",
    "cookie",
    "token",
    "auth_token",
    "anthropic_auth_token",
    "password",
    "secret",
}


def redact_payload(payload: Any, *, enabled: bool) -> Any:
    if not enabled:
        return payload
    return _redact(payload)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        # Keys are coerced to str so the redacted output is always JSON-safe;
        # mixed-type-key dicts collapse on string collisions, which is
        # acceptable since redacted records are for human/log inspection.
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower().replace("-", "_") in _REDACT_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact(item) for item in value]
    return value
