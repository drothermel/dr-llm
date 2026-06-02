from __future__ import annotations

from dr_llm.streaming_log.payloads import PayloadRef

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.models import ArtifactLane


ALWAYS_PROJECT_ROLES = frozenset(
    {
        "request_json",
        "response_json",
        "stdout",
        "stderr",
        "generated_code",
        "validation_report",
        "metrics_payload",
        "metrics",
        "error_detail",
    }
)


class ArtifactRolePolicy:
    def __init__(self, config: ArtifactProjectionConfig) -> None:
        self.config = config

    def should_project(self, payload_ref: PayloadRef) -> bool:
        if payload_ref.role in ALWAYS_PROJECT_ROLES:
            return True
        if payload_ref.role == "metadata_json":
            return (
                payload_ref.size_bytes >= self.config.metadata_spill_threshold
            )
        return False

    def lane_for(self, payload_ref: PayloadRef) -> ArtifactLane:
        if payload_ref.encoding == "binary":
            return ArtifactLane.binary
        if _is_json_content(payload_ref.content_type):
            return ArtifactLane.json
        return ArtifactLane.text


def _is_json_content(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


__all__ = ["ALWAYS_PROJECT_ROLES", "ArtifactRolePolicy"]
