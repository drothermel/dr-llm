from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
import os

import psycopg
from psycopg import sql
import pytest

from dr_llm.errors import TransientPersistenceError
from dr_llm.metadata_projection import (
    MetadataAssertion,
    MetadataAssertionRole,
    MetadataEntity,
    MetadataProjectionCheckpoint,
    MetadataProjectionConfig,
    MetadataStore,
    MetadataWritePlan,
    assertion_id,
    entity_id,
)
from dr_llm.pool.db.runtime import DbConfig, DbRuntime

pytestmark = pytest.mark.integration


METADATA_TABLES = (
    "metadata_assertion_roles",
    "metadata_assertions",
    "metadata_entities",
    "metadata_projection_checkpoints",
    "metadata_projection_errors",
)


@pytest.fixture()
def metadata_store() -> Generator[MetadataStore, None, None]:
    dsn = _dsn()
    _drop_metadata_tables(dsn)
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="metadata_projection_tests",
        )
    )
    try:
        store = MetadataStore(
            config=MetadataProjectionConfig(database_dsn=dsn),
            runtime=runtime,
        )
        store.initialize()
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        runtime.close()
        pytest.skip(f"Postgres unavailable for metadata tests: {exc}")
    yield store
    store.close()
    _drop_metadata_tables(dsn)


def test_store_replay_is_idempotent(metadata_store: MetadataStore) -> None:
    plan = _plan(["run"])
    checkpoint = MetadataProjectionCheckpoint(
        projection_version="metadata-v1",
        durable_consumer="drllm_metadata_projection_v1",
        stream_sequence=1,
        event_id="event-1",
    )

    metadata_store.apply_write_plan(plan, checkpoint=checkpoint)
    metadata_store.apply_write_plan(plan, checkpoint=checkpoint)

    summary = metadata_store.summary()
    assert summary.entity_count == 1
    assert summary.assertion_count == 1
    assert summary.role_count == 1
    assert summary.error_count == 0
    assert summary.checkpoint is not None
    assert summary.checkpoint.stream_sequence == 1


def test_conflicting_assertion_does_not_add_roles(
    metadata_store: MetadataStore,
) -> None:
    metadata_store.apply_write_plan(_plan(["run"]))

    metadata_store.apply_write_plan(_plan(["run", "work"]))

    summary = metadata_store.summary()
    assert summary.assertion_count == 1
    assert summary.role_count == 1
    assert summary.error_count == 1


def _plan(role_entity_types: list[str]) -> MetadataWritePlan:
    assertion = _assertion()
    entities = [_entity(entity_type) for entity_type in role_entity_types]
    return MetadataWritePlan(
        entities=entities,
        assertions=[assertion],
        roles=[
            MetadataAssertionRole(
                assertion_id=assertion.assertion_id,
                role_name=entity.entity_type,
                entity_id=entity.entity_id,
            )
            for entity in entities
        ],
    )


def _assertion() -> MetadataAssertion:
    return MetadataAssertion(
        assertion_id=assertion_id(
            projection_version="metadata-v1",
            assertion_type="work_submitted",
            source_idempotency_key="idem-1",
        ),
        assertion_type="work_submitted",
        projection_version="metadata-v1",
        source_event_id="event-1",
        source_event_type="work_submitted",
        source_schema_version=1,
        source_idempotency_key="idem-1",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _entity(entity_type: str) -> MetadataEntity:
    identity_key = f"{entity_type}-1"
    return MetadataEntity(
        entity_id=entity_id(entity_type, identity_key),
        entity_type=entity_type,
        identity_key=identity_key,
    )


def _dsn() -> str:
    dsn = os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv(
        "DR_LLM_DATABASE_URL"
    )
    if dsn is None:
        pytest.skip(
            "Set DR_LLM_TEST_DATABASE_URL to run metadata integration tests"
        )
    return dsn


def _drop_metadata_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for table_name in METADATA_TABLES:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(table_name)
                )
            )
        conn.commit()
