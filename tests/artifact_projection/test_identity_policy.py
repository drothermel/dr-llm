from __future__ import annotations

from typing import Literal

from dr_llm.artifact_projection import (
    ArtifactLane,
    ArtifactProjectionConfig,
    ArtifactRolePolicy,
    ArtifactSourceRef,
    PayloadArtifactSource,
    artifact_id_for_source_ref,
)
from dr_llm.streaming_log.payloads import PayloadRef


def test_artifact_id_is_stable_and_depends_on_logical_source() -> None:
    source = _source(payload_role="response_json")

    first = artifact_id_for_source_ref(
        projection_version="artifact-v1", source_ref=source.source_ref
    )
    second = artifact_id_for_source_ref(
        projection_version="artifact-v1", source_ref=source.source_ref
    )
    changed = artifact_id_for_source_ref(
        projection_version="artifact-v1",
        source_ref=_source(payload_role="request_json").source_ref,
    )

    assert first == second
    assert first.startswith("art_")
    assert first != changed


def test_role_policy_projects_raw_roles_and_thresholded_metadata() -> None:
    policy = ArtifactRolePolicy(
        ArtifactProjectionConfig(metadata_spill_threshold=16)
    )

    assert policy.should_project(_ref(role="response_json", size_bytes=1))
    assert policy.should_project(_ref(role="metadata_json", size_bytes=16))
    assert not policy.should_project(_ref(role="metadata_json", size_bytes=15))
    assert not policy.should_project(_ref(role="pool_schema", size_bytes=100))


def test_role_policy_selects_lane_from_content_type_and_encoding() -> None:
    policy = ArtifactRolePolicy(ArtifactProjectionConfig())

    assert (
        policy.lane_for(_ref(content_type="application/json"))
        is ArtifactLane.json
    )
    assert (
        policy.lane_for(_ref(content_type="application/problem+json"))
        is ArtifactLane.json
    )
    assert (
        policy.lane_for(_ref(content_type="text/plain")) is ArtifactLane.text
    )
    assert (
        policy.lane_for(_ref(content_type="image/png", encoding="binary"))
        is ArtifactLane.binary
    )


def _source(*, payload_role: str) -> PayloadArtifactSource:
    return PayloadArtifactSource(
        source_ref=ArtifactSourceRef(
            event_id="event-1",
            event_type="provider_response_received",
            schema_version=1,
            idempotency_key="idem-1",
            payload_role=payload_role,
            object_key="sha256/ab/abc",
            sha256="a" * 64,
            size_bytes=3,
            content_type="application/json",
            encoding="utf-8",
            compression="none",
        )
    )


def _ref(
    *,
    role: str = "response_json",
    size_bytes: int = 1,
    content_type: str = "application/json",
    encoding: Literal["utf-8", "binary"] = "utf-8",
) -> PayloadRef:
    return PayloadRef(
        role=role,
        object_key="sha256/ab/abc",
        sha256="a" * 64,
        size_bytes=size_bytes,
        content_type=content_type,
        encoding=encoding,
        compression="none",
    )
