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
from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.models import AcquireQuery, DeletePoolRequest, PoolDeletionStatus
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.project_info import ProjectInfo


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _project_for_dsn(name: str, dsn: str) -> ProjectInfo:
    parsed = urlparse(dsn)
    assert parsed.port is not None
    return ProjectInfo(name=name, port=parsed.port, status=ContainerStatus.RUNNING)


def _pool_table_names(pool_name: str) -> list[str]:
    return [
        f"pool_{pool_name}_samples",
        f"pool_{pool_name}_claims",
        f"pool_{pool_name}_pending",
        f"pool_{pool_name}_metadata",
        f"pool_{pool_name}_call_stats",
    ]


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


def _create_legacy_tables(dsn: str, pool_name: str, *, partial: bool = False) -> None:
    table_names = _pool_table_names(pool_name)
    create_statements: list[sql.Composed | sql.SQL] = [
        sql.SQL("CREATE TABLE {} (sample_id text, payload_json jsonb)").format(
            sql.Identifier("public", table_names[0])
        ),
        sql.SQL('CREATE TABLE {} ("key" text, value_json jsonb)').format(
            sql.Identifier("public", table_names[3])
        ),
    ]
    if not partial:
        create_statements[1:1] = [
            sql.SQL("CREATE TABLE {} (claim_id text)").format(
                sql.Identifier("public", table_names[1])
            ),
            sql.SQL("CREATE TABLE {} (status text)").format(
                sql.Identifier("public", table_names[2])
            ),
        ]
        create_statements.append(
            sql.SQL("CREATE TABLE {} (sample_id text)").format(
                sql.Identifier("public", table_names[4])
            )
        )
    with psycopg.connect(dsn) as conn:
        for statement in create_statements:
            conn.execute(statement)
        conn.execute(
            sql.SQL("INSERT INTO {} (sample_id, payload_json) VALUES (%s, %s)").format(
                sql.Identifier("public", table_names[0])
            ),
            ["sample-1", "{}"],
        )
        conn.execute(
            sql.SQL("INSERT INTO {} (key, value_json) VALUES (%s, %s)").format(
                sql.Identifier("public", table_names[3])
            ),
            ["legacy", "{}"],
        )
        if not partial:
            conn.execute(
                sql.SQL("INSERT INTO {} (claim_id) VALUES (%s)").format(
                    sql.Identifier("public", table_names[1])
                ),
                ["claim-1"],
            )
            conn.execute(
                sql.SQL("INSERT INTO {} (status) VALUES (%s)").format(
                    sql.Identifier("public", table_names[2])
                ),
                [PendingStatus.failed.value],
            )
            conn.execute(
                sql.SQL("INSERT INTO {} (sample_id) VALUES (%s)").format(
                    sql.Identifier("public", table_names[4])
                ),
                ["sample-1"],
            )
        conn.commit()


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
            schema.samples_table: 1,
            schema.claims_table: 1,
            schema.pending_table: 1,
            schema.metadata_table: 2,
            schema.call_stats_table: 1,
        }
        for table_name in _pool_table_names(pool_name):
            assert _table_exists(dsn, table_name) is False
    finally:
        runtime.close()
        _drop_tables(dsn, _pool_table_names(pool_name))


@pytest.mark.integration
def test_delete_pool_supports_legacy_pool_without_schema_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    pool_name = f"legacy_{uuid4().hex[:8]}"
    table_names = _pool_table_names(pool_name)

    try:
        _create_legacy_tables(dsn, pool_name)
        monkeypatch.setattr(
            admin_service,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = admin_service.delete_pool(
            DeletePoolRequest(project_name="demo", pool_name=pool_name)
        )

        assert result.status == PoolDeletionStatus.deleted
        assert result.pre_delete_counts == {
            table_names[0]: 1,
            table_names[1]: 1,
            table_names[2]: 1,
            table_names[3]: 1,
            table_names[4]: 1,
        }
        for table_name in table_names:
            assert _table_exists(dsn, table_name) is False
    finally:
        _drop_tables(dsn, table_names)


@pytest.mark.integration
def test_delete_pool_succeeds_when_ancillary_tables_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    pool_name = f"partial_{uuid4().hex[:8]}"
    table_names = _pool_table_names(pool_name)

    try:
        _create_legacy_tables(dsn, pool_name, partial=True)
        monkeypatch.setattr(
            admin_service,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = admin_service.delete_pool(
            DeletePoolRequest(project_name="demo", pool_name=pool_name)
        )

        assert result.status == PoolDeletionStatus.deleted
        assert result.deleted_table_names == [table_names[0], table_names[3]]
        assert result.missing_table_names == [
            table_names[1],
            table_names[2],
            table_names[4],
        ]
        for table_name in table_names:
            assert _table_exists(dsn, table_name) is False
    finally:
        _drop_tables(dsn, table_names)
