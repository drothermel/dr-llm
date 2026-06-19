from __future__ import annotations

import hashlib

from dr_llm.streaming_log.serialization import canonical_json_bytes

from dr_llm.artifact_projection.models import ArtifactSourceRef


def artifact_id_for_source_ref(
    *, projection_version: str, source_ref: ArtifactSourceRef
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "projection_version": projection_version,
                "source_idempotency_key": source_ref.idempotency_key,
                "payload_role": source_ref.payload_role,
                "source_object_key": source_ref.object_key,
                "source_sha256": source_ref.sha256,
            }
        )
    ).hexdigest()
    return f"art_{digest}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["artifact_id_for_source_ref", "sha256_bytes"]
