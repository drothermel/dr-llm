from __future__ import annotations

import hashlib
from typing import Any

from dr_llm.streaming_log.serialization import canonical_json_bytes


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def entity_id(entity_type: str, identity_key: str) -> str:
    return "ent_" + canonical_sha256(
        {"entity_type": entity_type, "identity_key": identity_key}
    )


def assertion_id(
    *,
    projection_version: str,
    assertion_type: str,
    source_idempotency_key: str,
) -> str:
    return "asrt_" + canonical_sha256(
        {
            "projection_version": projection_version,
            "assertion_type": assertion_type,
            "source_idempotency_key": source_idempotency_key,
        }
    )


def content_hash(value: Any) -> str:
    return canonical_sha256(value)


__all__ = [
    "assertion_id",
    "canonical_sha256",
    "content_hash",
    "entity_id",
]
