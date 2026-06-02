from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.identity import (
    artifact_id_for_source_ref,
    sha256_bytes,
)
from dr_llm.artifact_projection.models import (
    PayloadArtifactSource,
    ProjectionCheckpoint,
    ProjectionError,
    ProjectionErrorKind,
)
from dr_llm.artifact_projection.policy import ArtifactRolePolicy
from dr_llm.artifact_projection.store import (
    ArtifactStore,
    projection_error_for_source,
)
from dr_llm.streaming_log.client import (
    StreamingLogConnection,
    StreamingPayloadReader,
    StreamingPayloadStore,
)
from dr_llm.streaming_log.events import EventEnvelope
from dr_llm.streaming_log.payloads import PayloadRef


class ArtifactProjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projected_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    error_count: int = Field(ge=0)


class ArtifactEventDelivery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    event: EventEnvelope
    stream_sequence: int = Field(ge=0)
    message: Any

    async def ack(self) -> None:
        await self.message.ack()


class ArtifactProjector:
    def __init__(
        self,
        *,
        config: ArtifactProjectionConfig,
        store: ArtifactStore,
        payload_reader: StreamingPayloadReader,
        role_policy: ArtifactRolePolicy | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.payload_reader = payload_reader
        self.role_policy = role_policy or ArtifactRolePolicy(config)

    async def process_delivery(
        self, delivery: ArtifactEventDelivery
    ) -> ArtifactProjectionResult:
        result = await self.process_event(
            delivery.event, stream_sequence=delivery.stream_sequence
        )
        self._record_checkpoint(delivery)
        await delivery.ack()
        return result

    async def process_event(
        self, event: EventEnvelope, *, stream_sequence: int
    ) -> ArtifactProjectionResult:
        projected_count = 0
        skipped_count = 0
        error_count = 0
        for payload_ref in event.payload_refs:
            if not self.role_policy.should_project(payload_ref):
                skipped_count += 1
                continue
            source = PayloadArtifactSource.from_event_ref(
                event=event, payload_ref=payload_ref
            )
            outcome = await self._project_source(
                source=source,
                payload_ref=payload_ref,
                stream_sequence=stream_sequence,
            )
            projected_count += int(outcome == "projected")
            skipped_count += int(outcome == "skipped")
            error_count += int(outcome == "error")
        self.store.finalize()
        return ArtifactProjectionResult(
            projected_count=projected_count,
            skipped_count=skipped_count,
            error_count=error_count,
        )

    async def _project_source(
        self,
        *,
        source: PayloadArtifactSource,
        payload_ref: PayloadRef,
        stream_sequence: int,
    ) -> str:
        artifact_id = artifact_id_for_source_ref(
            projection_version=self.config.projection_version,
            source_ref=source.source_ref,
        )
        if self.store.existing_reference(artifact_id=artifact_id) is not None:
            return "skipped"
        if source.source_ref.compression != "none":
            self._record_source_error(
                source=source,
                error_kind=ProjectionErrorKind.unsupported_source_compression,
                message=(
                    "Unsupported source compression "
                    f"{source.source_ref.compression!r}"
                ),
                stream_sequence=stream_sequence,
            )
            return "error"
        data = await self._read_source_payload(
            source=source,
            payload_ref=payload_ref,
            stream_sequence=stream_sequence,
        )
        if data is None:
            return "error"
        lane = self.role_policy.lane_for(payload_ref)
        self.store.write_artifact(source=source, lane=lane, data=data)
        return "projected"

    async def _read_source_payload(
        self,
        *,
        source: PayloadArtifactSource,
        payload_ref: PayloadRef,
        stream_sequence: int,
    ) -> bytes | None:
        try:
            data = await self.payload_reader.read_payload_ref(payload_ref)
        except Exception as exc:  # noqa: BLE001
            self._record_source_error(
                source=source,
                error_kind=ProjectionErrorKind.missing_payload,
                message=str(exc),
                stream_sequence=stream_sequence,
            )
            return None
        return self._verified_payload_bytes(
            source=source, data=data, stream_sequence=stream_sequence
        )

    def _verified_payload_bytes(
        self,
        *,
        source: PayloadArtifactSource,
        data: bytes,
        stream_sequence: int,
    ) -> bytes | None:
        if len(data) != source.source_ref.size_bytes:
            self._record_source_error(
                source=source,
                error_kind=ProjectionErrorKind.source_size_mismatch,
                message=(
                    f"Expected {source.source_ref.size_bytes} bytes, "
                    f"read {len(data)} bytes"
                ),
                stream_sequence=stream_sequence,
            )
            return None
        digest = sha256_bytes(data)
        if digest != source.source_ref.sha256:
            self._record_source_error(
                source=source,
                error_kind=ProjectionErrorKind.source_hash_mismatch,
                message=(
                    f"Expected sha256 {source.source_ref.sha256}, read {digest}"
                ),
                stream_sequence=stream_sequence,
            )
            return None
        return data

    def _record_source_error(
        self,
        *,
        source: PayloadArtifactSource,
        error_kind: ProjectionErrorKind,
        message: str,
        stream_sequence: int,
    ) -> ProjectionError:
        error = projection_error_for_source(
            config=self.config,
            source=source,
            error_kind=error_kind,
            message=message,
            stream_sequence=stream_sequence,
        )
        self.store.record_error(error)
        return error

    def _record_checkpoint(self, delivery: ArtifactEventDelivery) -> None:
        self.store.index.record_checkpoint(
            ProjectionCheckpoint(
                projection_version=self.config.projection_version,
                durable_consumer=self.config.durable_consumer,
                stream_sequence=delivery.stream_sequence,
                event_id=delivery.event.event_id,
            )
        )


async def run_artifact_projector(
    *,
    connection: StreamingLogConnection,
    config: ArtifactProjectionConfig,
    max_messages: int | None = None,
    batch_size: int | None = None,
    flush_on_exit: bool = True,
) -> int:
    store = ArtifactStore(config=config)
    store.initialize()
    payload_reader = StreamingPayloadStore(connection)
    projector = ArtifactProjector(
        config=config,
        store=store,
        payload_reader=payload_reader,
    )
    try:
        return await _consume_events(
            connection=connection,
            config=config,
            projector=projector,
            max_messages=max_messages,
            batch_size=batch_size or config.fetch_batch_size,
        )
    finally:
        if flush_on_exit:
            store.finalize()


async def _consume_events(
    *,
    connection: StreamingLogConnection,
    config: ArtifactProjectionConfig,
    projector: ArtifactProjector,
    max_messages: int | None,
    batch_size: int,
) -> int:
    sub = await connection.js.pull_subscribe(
        connection.config.events_subject,
        durable=config.durable_consumer,
        stream=connection.config.events_stream,
    )
    processed = 0
    while max_messages is None or processed < max_messages:
        messages = await sub.fetch(batch_size, timeout=1)
        for message in messages:
            delivery = delivery_from_message(message)
            await projector.process_delivery(delivery)
            processed += 1
            if max_messages is not None and processed >= max_messages:
                break
    return processed


def delivery_from_message(message: Any) -> ArtifactEventDelivery:
    return ArtifactEventDelivery(
        event=EventEnvelope.model_validate_json(message.data),
        stream_sequence=stream_sequence_for_message(message),
        message=message,
    )


def stream_sequence_for_message(message: Any) -> int:
    metadata = getattr(message, "metadata", None)
    sequence = getattr(metadata, "sequence", None)
    stream_sequence = getattr(sequence, "stream", None)
    if stream_sequence is None:
        raise ValueError("message metadata is missing stream sequence")
    return int(stream_sequence)


__all__ = [
    "ArtifactEventDelivery",
    "ArtifactProjectionResult",
    "ArtifactProjector",
    "delivery_from_message",
    "run_artifact_projector",
    "stream_sequence_for_message",
]
