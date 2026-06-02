from __future__ import annotations

import hashlib

from dr_llm.streaming_log.serialization import canonical_json_bytes

from dr_llm.artifact_projection.models import PayloadArtifactSource


def artifact_id_for_source(
    *, projection_version: str, source: PayloadArtifactSource
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "projection_version": projection_version,
                "source_idempotency_key": source.source_idempotency_key,
                "payload_role": source.payload_role,
                "source_object_key": source.source_object_key,
                "source_sha256": source.source_sha256,
            }
        )
    ).hexdigest()
    return f"art_{digest}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["artifact_id_for_source", "sha256_bytes"]
