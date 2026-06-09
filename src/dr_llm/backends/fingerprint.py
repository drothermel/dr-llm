"""Canonical request fingerprinting for pool cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from dr_llm.backends.models import BackendRequest


def fingerprint_request(request: BackendRequest) -> str:
    """Return a stable SHA-256 hex fingerprint for cache and pool keys."""
    canonical = _canonical_payload(request)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_payload(request: BackendRequest) -> dict[str, Any]:
    payload = request.model_dump(
        mode="json",
        include={
            "provider",
            "model",
            "mode",
            "messages",
            "max_tokens",
            "effort",
            "reasoning",
            "sampling",
        },
        exclude_none=False,
    )
    return payload
