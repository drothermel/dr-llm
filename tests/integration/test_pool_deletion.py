"""Integration tests for pool deletion (requires PostgreSQL)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.admin import deletion
from dr_llm.pool.admin.deletion import DeletePoolRequest, PoolDeletionStatus
from dr_llm.pool.db.names import PoolTableType
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import KeyColumn, PoolSchema, pool_table_names
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project import project_service as project_service_module


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


def _drop_tables(dsn: str, table_names: Iterable[str]) -> None:
    with psycopg.connect(dsn) as conn:
        for table_name in table_names:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", table_name)
                )
            )
        conn.commit()


def _drop_pool(dsn: str, pool_name: str) -> None:
    _drop_tables(dsn, pool_table_names(pool_name))
    with psycopg.connect(dsn) as conn:
        conn.execute("DELETE FROM pool_catalog WHERE pool_name = %s", [pool_name])
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


def _catalog_entry_exists(dsn: str, pool_name: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pool_catalog WHERE pool_name = %s)",
            [pool_name],
        ).fetchone()
    assert row is not None
    return bool(row[0])


@pytest.mark.integration
def test_delete_pool_reports_counts_for_normal_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    pool_name = f"delete_{uuid4().hex[:8]}"
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    schema = PoolSchema(name=pool_name, key_columns=[KeyColumn(name="dim_a")])
    store = PoolStore(schema, runtime)

    try:
        try:
            store.ensure_schema()
        except (psycopg.OperationalError, TransientPersistenceError) as exc:
            pytest.skip(f"Postgres unavailable for pool integration tests: {exc}")

        store.insert_sample(
            PoolSample(
                key_values={"dim_a": "alpha"},
                sample_idx=0,
                request={"prompt": "hello"},
            )
        )
        store.insert_sample(
            PoolSample(
                key_values={"dim_a": "beta"},
                sample_idx=0,
                request={"prompt": "world"},
            )
        )

        monkeypatch.setattr(
            project_service_module,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = deletion.delete_pool(
            DeletePoolRequest(project_name="demo", pool_name=pool_name)
        )

        assert result.status == PoolDeletionStatus.deleted
        assert result.deleted_table_names == pool_table_names(pool_name)
        assert result.pre_delete_counts == {
            schema.table_name(PoolTableType.SAMPLES): 2,
            schema.table_name(PoolTableType.LEASES): 0,
        }
        for table_name in pool_table_names(pool_name):
            assert _table_exists(dsn, table_name) is False
        assert _catalog_entry_exists(dsn, pool_name) is False
    finally:
        runtime.close()
        _drop_pool(dsn, pool_name)


@pytest.mark.integration
def test_delete_pool_with_active_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    pool_name = f"del_lease_{uuid4().hex[:8]}"
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    schema = PoolSchema(name=pool_name, key_columns=[KeyColumn(name="dim_a")])
    store = PoolStore(schema, runtime)

    try:
        try:
            store.ensure_schema()
        except (psycopg.OperationalError, TransientPersistenceError) as exc:
            pytest.skip(f"Postgres unavailable for pool integration tests: {exc}")

        store.insert_sample(
            PoolSample(
                key_values={"dim_a": "alpha"},
                sample_idx=0,
                request={"prompt": "hello"},
            )
        )
        store.insert_sample(
            PoolSample(
                key_values={"dim_a": "beta"},
                sample_idx=0,
                request={"prompt": "world"},
            )
        )
        store.claim_lease(worker_id="w1", lease_seconds=600)

        monkeypatch.setattr(
            project_service_module,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = deletion.delete_pool(
            DeletePoolRequest(project_name="demo", pool_name=pool_name)
        )

        assert result.status == PoolDeletionStatus.deleted
        assert result.pre_delete_counts[schema.table_name(PoolTableType.SAMPLES)] == 2
        assert result.pre_delete_counts[schema.table_name(PoolTableType.LEASES)] == 1
        for table_name in pool_table_names(pool_name):
            assert _table_exists(dsn, table_name) is False
        assert _catalog_entry_exists(dsn, pool_name) is False
    finally:
        runtime.close()
        _drop_pool(dsn, pool_name)
