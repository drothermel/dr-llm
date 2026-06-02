from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.db import DbConfig, DbRuntime
from dr_llm.pool.db.catalog import list_pool_names
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.reader import PoolReader
from dr_llm.streaming_log.client import StreamingEventLog
from dr_llm.streaming_log.event_builders import (
    StreamingEventPublishSpec,
    pool_sample_imported_event,
)
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    PoolImportCompletedPayload,
    PoolImportFailedPayload,
    PoolImportStartedPayload,
    StreamingLogEventType,
    idempotency_key,
)
from dr_llm.streaming_log.payloads import prepare_json_payload


logger = logging.getLogger(__name__)


class PoolImportResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    imported_count: int = 0
    event_ids: list[str] = Field(default_factory=list)


class PoolSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    source_id: str
    schema_payload: dict[str, Any]


class PoolSnapshotSource:
    def __init__(
        self,
        *,
        dsn: str,
        pool_name: str,
        source_id: str | None = None,
        sample_limit: int | None = None,
    ) -> None:
        self.dsn = dsn
        self.pool_name = pool_name
        self.source_id = source_id or dsn
        self.sample_limit = sample_limit
        self._runtime: DbRuntime | None = None
        self._reader: PoolReader | None = None
        self._snapshot: PoolSnapshot | None = None

    def __enter__(self) -> PoolSnapshotSource:
        self._runtime = _db_runtime(
            dsn=self.dsn,
            application_name="dr_llm_streaming_log_import",
        )
        try:
            self._reader = PoolReader.open(
                self.pool_name, runtime=self._runtime
            )
            self._snapshot = PoolSnapshot(
                pool_name=self.pool_name,
                source_id=self.source_id,
                schema_payload=_schema_payload(self._reader),
            )
        except Exception:  # noqa: BLE001
            self.close()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def snapshot(self) -> PoolSnapshot:
        if self._snapshot is None:
            raise RuntimeError("pool snapshot source is not open")
        return self._snapshot

    def samples(self) -> Iterator[PoolSample]:
        if self._reader is None:
            raise RuntimeError("pool snapshot source is not open")
        for imported_count, sample in enumerate(self._reader.samples()):
            if (
                self.sample_limit is not None
                and imported_count >= self.sample_limit
            ):
                break
            yield sample

    def close(self) -> None:
        if self._runtime is None:
            return
        self._runtime.close()
        self._runtime = None
        self._reader = None
        self._snapshot = None


class PoolImportEventRecorder:
    def __init__(
        self,
        *,
        event_log: StreamingEventLog,
        pool_name: str,
        source_id: str,
        schema_payload: dict[str, Any] | None = None,
    ) -> None:
        self.event_log = event_log
        self.pool_name = pool_name
        self.source_id = source_id
        self.schema_payload = schema_payload
        self.import_context = EventContext(source=source_id)

    @classmethod
    def from_snapshot(
        cls, *, event_log: StreamingEventLog, snapshot: PoolSnapshot
    ) -> PoolImportEventRecorder:
        return cls(
            event_log=event_log,
            pool_name=snapshot.pool_name,
            source_id=snapshot.source_id,
            schema_payload=snapshot.schema_payload,
        )

    async def record_started(self) -> EventEnvelope:
        if self.schema_payload is None:
            raise ValueError(
                "pool schema payload is required for start events"
            )
        return await self.event_log.publish_event_spec(
            StreamingEventPublishSpec(
                event_type=StreamingLogEventType.pool_import_started,
                idempotency_key=idempotency_key(
                    self.source_id, self.pool_name, "pool_import_started"
                ),
                payload=PoolImportStartedPayload(
                    pool_name=self.pool_name,
                    source_id=self.source_id,
                ),
                payloads=[
                    prepare_json_payload("pool_schema", self.schema_payload)
                ],
                context=self.import_context,
            )
        )

    async def record_sample_imported(
        self, sample: PoolSample
    ) -> EventEnvelope:
        if self.schema_payload is None:
            raise ValueError(
                "pool schema payload is required for sample events"
            )
        return await self.event_log.publish_event_spec(
            pool_sample_imported_event(
                pool_name=self.pool_name,
                source_id=self.source_id,
                schema_payload=self.schema_payload,
                sample=sample,
            )
        )

    async def record_completed(self, imported_count: int) -> EventEnvelope:
        return await self.event_log.publish_event_spec(
            StreamingEventPublishSpec(
                event_type=StreamingLogEventType.pool_import_completed,
                idempotency_key=idempotency_key(
                    self.source_id,
                    self.pool_name,
                    "pool_import_completed",
                    imported_count,
                ),
                payload=PoolImportCompletedPayload(
                    pool_name=self.pool_name,
                    source_id=self.source_id,
                    imported_count=imported_count,
                    reconstructed=True,
                ),
                context=self.import_context,
            )
        )

    async def record_failed(self, exc: Exception) -> EventEnvelope:
        return await self.event_log.publish_event_spec(
            StreamingEventPublishSpec(
                event_type=StreamingLogEventType.pool_import_failed,
                idempotency_key=idempotency_key(
                    self.source_id,
                    self.pool_name,
                    "pool_import_failed",
                    type(exc).__name__,
                ),
                payload=PoolImportFailedPayload(
                    pool_name=self.pool_name,
                    source_id=self.source_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                ),
                context=self.import_context,
            )
        )


async def ingest_pool(
    *,
    event_log: StreamingEventLog,
    dsn: str,
    pool_name: str,
    source_id: str | None = None,
    sample_limit: int | None = None,
) -> PoolImportResult:
    _validate_sample_limit(sample_limit)
    source = PoolSnapshotSource(
        dsn=dsn,
        pool_name=pool_name,
        source_id=source_id,
        sample_limit=sample_limit,
    )
    source_opened = False
    try:
        with source:
            source_opened = True
            return await record_pool_import(
                event_log=event_log,
                snapshot=source.snapshot,
                samples=source.samples(),
            )
    except Exception as exc:  # noqa: BLE001
        if not source_opened:
            recorder = PoolImportEventRecorder(
                event_log=event_log,
                pool_name=pool_name,
                source_id=source_id or dsn,
            )
            await _record_source_failure(recorder=recorder, exc=exc)
        raise


async def record_pool_import(
    *,
    event_log: StreamingEventLog,
    snapshot: PoolSnapshot,
    samples: Iterable[PoolSample],
) -> PoolImportResult:
    recorder = PoolImportEventRecorder.from_snapshot(
        event_log=event_log, snapshot=snapshot
    )
    event_ids: list[str] = []
    imported_count = 0
    started = await recorder.record_started()
    event_ids.append(started.event_id)
    sample_iterator = iter(samples)
    while True:
        try:
            sample = next(sample_iterator)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001
            await _record_source_failure(recorder=recorder, exc=exc)
            raise
        event = await recorder.record_sample_imported(sample)
        event_ids.append(event.event_id)
        imported_count += 1
    completed = await recorder.record_completed(imported_count)
    event_ids.append(completed.event_id)
    return PoolImportResult(
        pool_name=snapshot.pool_name,
        imported_count=imported_count,
        event_ids=event_ids,
    )


async def _record_source_failure(
    *, recorder: PoolImportEventRecorder, exc: Exception
) -> None:
    try:
        await recorder.record_failed(exc)
    except Exception as failure_event_exc:  # noqa: BLE001
        exc.add_note(
            "Publishing pool_import_failed event also failed: "
            f"{type(failure_event_exc).__name__}: {failure_event_exc}"
        )
        logger.exception(
            "Failed to publish pool_import_failed event for pool %r "
            "from source %r after %s: %s",
            recorder.pool_name,
            recorder.source_id,
            type(exc).__name__,
            exc,
        )


def _db_runtime(*, dsn: str, application_name: str) -> DbRuntime:
    return DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=4,
            application_name=application_name,
        )
    )


async def ingest_pools(
    *,
    event_log: StreamingEventLog,
    dsn: str,
    source_id: str | None = None,
    sample_limit: int | None = None,
) -> list[PoolImportResult]:
    _validate_sample_limit(sample_limit)
    runtime = _db_runtime(
        dsn=dsn,
        application_name="dr_llm_streaming_log_import_discovery",
    )
    try:
        pool_names = list_pool_names(runtime)
    finally:
        runtime.close()
    results: list[PoolImportResult] = []
    for pool_name in pool_names:
        results.append(
            await ingest_pool(
                event_log=event_log,
                dsn=dsn,
                pool_name=pool_name,
                source_id=source_id,
                sample_limit=sample_limit,
            )
        )
    return results


def _schema_payload(reader: PoolReader) -> dict[str, Any]:
    return reader.schema.model_dump(mode="json", exclude_computed_fields=True)


def _validate_sample_limit(sample_limit: int | None) -> None:
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be at least 1 when provided")


__all__ = [
    "PoolImportEventRecorder",
    "PoolImportResult",
    "PoolSnapshot",
    "PoolSnapshotSource",
    "ingest_pool",
    "ingest_pools",
    "record_pool_import",
]
