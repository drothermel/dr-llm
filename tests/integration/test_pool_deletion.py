from __future__ import annotations

import os
from collections.abc import Iterable
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

import dr_llm.pool.admin_service as admin_service
from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.call_stats import CallStats
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import (
    KeyColumn,
    PoolSchema,
    PoolTableType,
    pool_table_names,
)
from dr_llm.pool.models import AcquireQuery, DeletePoolRequest, PoolDeletionStatus
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.project_info import ProjectInfo


def _get_dsn() -> str:
    dsn = os.getenv("DR_LLM_TEST_DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "Set DR_LLM_TEST_DATABASE_URL to run destructive pool integration tests"
        )
    return dsn


def _project_for_dsn(name: str, dsn: str) -> ProjectInfo:
    parsed = urlparse(dsn)
    assert parsed.port is not None
    return ProjectInfo(name=name, port=parsed.port, status=ContainerStatus.RUNNING)


def _pool_table_names(pool_name: str) -> list[str]:
    return pool_table_names(pool_name)


def _drop_tables(dsn: str, table_names: Iterable[str]) -> None:
    with psycopg.connect(dsn) as conn:
        for table_name in table_names:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", table_name)
                )
            )
        conn.commit()


def _table_exists(dsn: str, table_name: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            [table_name],
        ).fetchone()
    assert row is not None
    return bool(row[0])


@pytest.mark.integration
def test_delete_pool_reports_counts_for_normal_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    pool_name = f"delete_{uuid4().hex[:8]}"
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    schema = PoolSchema(name=pool_name, key_columns=[KeyColumn(name="dim_a")])
    store = PoolStore(schema, runtime)
    sample = PoolSample(
        key_values={"dim_a": "alpha"},
        sample_idx=0,
    )

    try:
        try:
            store.ensure_schema()
        except (psycopg.OperationalError, TransientPersistenceError) as exc:
            pytest.skip(f"Postgres unavailable for pool integration tests: {exc}")
        store.insert_sample(sample)
        store.acquire(AcquireQuery(run_id="run-1", key_values={"dim_a": "alpha"}, n=1))
        store.pending.insert(
            PendingSample(
                key_values={"dim_a": "alpha"},
                sample_idx=1,
                status=PendingStatus.failed,
            )
        )
        store.metadata.upsert("extra", {"source": "test"})
        store.insert_call_stats(
            CallStats(
                sample_id=sample.sample_id,
                latency_ms=12,
                total_cost_usd=0.5,
                prompt_tokens=10,
                completion_tokens=5,
                reasoning_tokens=0,
                total_tokens=15,
                attempt_count=1,
                finish_reason="stop",
            )
        )

        monkeypatch.setattr(
            admin_service,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = admin_service.delete_pool(
            DeletePoolRequest(project_name="demo", pool_name=pool_name)
        )

        assert result.status == PoolDeletionStatus.deleted
        assert result.deleted_table_names == _pool_table_names(pool_name)
        assert result.pre_delete_counts == {
            schema.table_name(PoolTableType.samples): 1,
            schema.table_name(PoolTableType.claims): 1,
            schema.table_name(PoolTableType.pending): 1,
            schema.table_name(PoolTableType.metadata): 2,
            schema.table_name(PoolTableType.call_stats): 1,
        }
        for table_name in _pool_table_names(pool_name):
            assert _table_exists(dsn, table_name) is False
    finally:
        runtime.close()
        _drop_tables(dsn, _pool_table_names(pool_name))


@pytest.mark.integration
def test_delete_pool_allows_pending_and_leased_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    pool_name = f"pending_{uuid4().hex[:8]}"
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    schema = PoolSchema(name=pool_name, key_columns=[KeyColumn(name="dim_a")])
    store = PoolStore(schema, runtime)

    try:
        try:
            store.ensure_schema()
        except (psycopg.OperationalError, TransientPersistenceError) as exc:
            pytest.skip(f"Postgres unavailable for pool integration tests: {exc}")
        store.pending.insert(
            PendingSample(
                key_values={"dim_a": "alpha"},
                sample_idx=0,
                status=PendingStatus.pending,
            )
        )
        store.pending.insert(
            PendingSample(
                key_values={"dim_a": "beta"},
                sample_idx=1,
                status=PendingStatus.leased,
            )
        )

        monkeypatch.setattr(
            admin_service,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = admin_service.delete_pool(
            DeletePoolRequest(project_name="demo", pool_name=pool_name)
        )

        assert result.status == PoolDeletionStatus.deleted
        assert result.pre_delete_counts[schema.table_name(PoolTableType.pending)] == 2
        for table_name in _pool_table_names(pool_name):
            assert _table_exists(dsn, table_name) is False
    finally:
        runtime.close()
        _drop_tables(dsn, _pool_table_names(pool_name))
