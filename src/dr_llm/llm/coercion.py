"""Tolerant value coercion helpers shared by catalog fetchers and providers.

These return ``None`` (or skip) on failure rather than raising, since they are
designed to ingest untyped JSON payloads from third-party APIs.
"""

from __future__ import annotations

from typing import Any


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    # int/float only (not bool; bool subclasses int but is handled above).
    if type(value) is int or type(value) is float:
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    return None


def as_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def first_float(*values: Any) -> float | None:
    for value in values:
        parsed = as_float(value)
        if parsed is not None:
            return parsed
    return None


def first_str(*values: Any) -> str | None:
    for value in values:
        parsed = as_str(value)
        if parsed is not None:
            return parsed
    return None


def first_bool(*values: Any) -> bool | None:
    for value in values:
        parsed = as_bool(value)
        if parsed is not None:
            return parsed
    return None
