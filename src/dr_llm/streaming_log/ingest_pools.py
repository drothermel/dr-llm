from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.db import DbConfig, DbRuntime
from dr_llm.pool.db.catalog import list_pool_names
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.reader import PoolReader
from dr_llm.streaming_log.client import StreamingLogClient
from dr_llm.streaming_log.events import (
    EventContext,
    StreamingLogEventType,
    idempotency_key,
    stable_hash,
)
from dr_llm.streaming_log.payloads import prepare_json_payload


class PoolImportResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    imported_count: int = 0
    event_ids: list[str] = Field(default_factory=list)
    failed: bool = False


async def ingest_pool(
    *,
    client: StreamingLogClient,
    dsn: str,
    pool_name: str,
    source_id: str | None = None,
    sample_limit: int | None = None,
) -> PoolImportResult:
    _validate_sample_limit(sample_limit)
    source = source_id or dsn
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=4,
            application_name="dr_llm_streaming_log_import",
        )
    )
    event_ids: list[str] = []
    imported_count = 0
    import_context = EventContext(source=source)
    try:
        reader = PoolReader.open(pool_name, runtime=runtime)
        schema_payload = reader.schema.model_dump(
            mode="json", exclude_computed_fields=True
        )
        started = await client.publish_event_with_payloads(
            StreamingLogEventType.pool_import_started,
            idempotency_key=idempotency_key(
                source, pool_name, "pool_import_started"
            ),
            payload={"pool_name": pool_name, "source_id": source},
            payloads=[prepare_json_payload("pool_schema", schema_payload)],
            context=import_context,
        )
        event_ids.append(started.event_id)
        for sample in reader.samples():
            if sample_limit is not None and imported_count >= sample_limit:
                break
            event = await _publish_sample_imported(
                client=client,
                sample=sample,
                pool_name=pool_name,
                schema=schema_payload,
                source=source,
            )
            event_ids.append(event.event_id)
            imported_count += 1
        completed = await client.publish_event_with_payloads(
            StreamingLogEventType.pool_import_completed,
            idempotency_key=idempotency_key(
                source,
                pool_name,
                "pool_import_completed",
                imported_count,
            ),
            payload={
                "pool_name": pool_name,
                "source_id": source,
                "imported_count": imported_count,
                "reconstructed": True,
            },
            context=import_context,
        )
        event_ids.append(completed.event_id)
        return PoolImportResult(
            pool_name=pool_name,
            imported_count=imported_count,
            event_ids=event_ids,
        )
    except Exception as exc:  # noqa: BLE001
        failed = await client.publish_event_with_payloads(
            StreamingLogEventType.pool_import_failed,
            idempotency_key=idempotency_key(
                source, pool_name, "pool_import_failed", type(exc).__name__
            ),
            payload={
                "pool_name": pool_name,
                "source_id": source,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            context=import_context,
        )
        event_ids.append(failed.event_id)
        raise
    finally:
        runtime.close()


async def ingest_pools(
    *,
    client: StreamingLogClient,
    dsn: str,
    source_id: str | None = None,
    sample_limit: int | None = None,
) -> list[PoolImportResult]:
    _validate_sample_limit(sample_limit)
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=4,
            application_name="dr_llm_streaming_log_import_discovery",
        )
    )
    try:
        pool_names = list_pool_names(runtime)
    finally:
        runtime.close()
    results: list[PoolImportResult] = []
    for pool_name in pool_names:
        results.append(
            await ingest_pool(
                client=client,
                dsn=dsn,
                pool_name=pool_name,
                source_id=source_id,
                sample_limit=sample_limit,
            )
        )
    return results


async def _publish_sample_imported(
    *,
    client: StreamingLogClient,
    sample: PoolSample,
    pool_name: str,
    schema: dict[str, Any],
    source: str,
):
    sample_payload = _sample_snapshot_payload(sample)
    state_hash = stable_hash(sample_payload)
    return await client.publish_event_with_payloads(
        StreamingLogEventType.pool_sample_imported,
        idempotency_key=idempotency_key(
            source,
            pool_name,
            sample.sample_id,
            sample.sample_idx,
            state_hash,
        ),
        payload={
            "pool_name": pool_name,
            "source_id": source,
            "sample_id": sample.sample_id,
            "sample_idx": sample.sample_idx,
            "run_id": sample.run_id,
            "key_values": sample.key_values,
            "finish_reason": sample.finish_reason,
            "attempt_count": sample.attempt_count,
            "created_at": (
                sample.created_at.isoformat()
                if sample.created_at is not None
                else None
            ),
            "completion_state": (
                "complete" if sample.is_complete else "incomplete"
            ),
            "reconstructed": True,
            "row_state_hash": state_hash,
        },
        payloads=[
            prepare_json_payload("pool_schema", schema),
            prepare_json_payload("request_json", sample.request),
            prepare_json_payload("metadata_json", sample.metadata),
            *(
                []
                if sample.response is None
                else [prepare_json_payload("response_json", sample.response)]
            ),
        ],
        context=EventContext(run_id=sample.run_id, source=source),
        metadata={"reconstructed": True},
    )


def _sample_snapshot_payload(sample: PoolSample) -> dict[str, Any]:
    return sample.model_dump(
        mode="json",
        exclude_none=True,
        exclude_computed_fields=True,
    )


def _validate_sample_limit(sample_limit: int | None) -> None:
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be at least 1 when provided")


__all__ = ["PoolImportResult", "ingest_pool", "ingest_pools"]
