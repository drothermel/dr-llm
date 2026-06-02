from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

from dr_llm.artifact_projection import (
    ArtifactProjectionConfig,
    ArtifactReader,
    ArtifactStore,
)
from dr_llm.artifact_projection.projector import run_artifact_projector
from dr_llm.streaming_log.bootstrap import bootstrap_streaming_log
from dr_llm.streaming_log.client import (
    StreamingEventLog,
    StreamingLogConnection,
    StreamingPayloadStore,
)
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.event_builders import StreamingEventPublishSpec
from dr_llm.streaming_log.events import (
    ProviderResponseReceivedPayload,
    StreamingLogEventType,
)
from dr_llm.streaming_log.payloads import prepare_json_payload

pytestmark = pytest.mark.integration


def _nats_url() -> str:
    value = os.getenv("DR_LLM_TEST_NATS_URL")
    if value is None:
        pytest.skip("Set DR_LLM_TEST_NATS_URL to run NATS integration tests")
    return value


def test_projector_reads_stream_payload_and_writes_artifact(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_projector_reads_stream_payload_and_writes_artifact(tmp_path)
    )


async def _test_projector_reads_stream_payload_and_writes_artifact(
    tmp_path: Path,
) -> None:
    streaming_config = _streaming_config()
    artifact_config = ArtifactProjectionConfig(artifact_root=tmp_path)
    await bootstrap_streaming_log(streaming_config)

    async with StreamingLogConnection(streaming_config) as connection:
        payload_store = StreamingPayloadStore(connection)
        event_log = StreamingEventLog(connection, payload_store)
        payload = prepare_json_payload("response_json", {"ok": True})
        await event_log.publish_event_spec(
            StreamingEventPublishSpec(
                event_type=StreamingLogEventType.provider_response_received,
                idempotency_key="artifact-integration-1",
                payload=ProviderResponseReceivedPayload(
                    provider="test",
                    model="test-model",
                    mode="api",
                ),
                payloads=[payload, payload],
            )
        )

        processed = await run_artifact_projector(
            connection=connection,
            config=artifact_config,
            max_messages=1,
        )

    store = ArtifactStore(config=artifact_config)
    references = store.index.list_references()
    summary = store.index.summary(
        projection_version=artifact_config.projection_version,
        durable_consumer=artifact_config.durable_consumer,
    )

    assert processed == 1
    assert len(references) == 1
    assert summary.artifact_count == 1
    assert summary.open_artifact_count == 0
    assert ArtifactReader(artifact_config).read_json(references[0]) == {
        "ok": True
    }


def _streaming_config() -> StreamingLogConfig:
    suffix = uuid4().hex[:8].upper()
    return StreamingLogConfig(
        nats_url=_nats_url(),
        events_stream=f"DRLLM_EVENTS_{suffix}",
        work_stream=f"DRLLM_WORK_{suffix}",
        payload_bucket=f"DRLLM_PAYLOADS_{suffix}",
        events_subject=f"drllm.events.{suffix.lower()}.>",
        work_subject=f"drllm.work.{suffix.lower()}.>",
        llm_work_subject=f"drllm.work.{suffix.lower()}.llm",
        work_consumer=f"drllm_work_workers_{suffix.lower()}",
        event_consumer=f"drllm_events_replay_{suffix.lower()}",
    )
