"""Integration tests for pool catalog backfill (requires PostgreSQL)."""

from __future__ import annotations

import os
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.admin.migration import (
    BackfillProjectCatalogRequest,
    PoolBackfillStatus,
    PoolMigrationError,
    backfill_project_catalog,
    derive_pool_schema,
)
from dr_llm.pool.db.catalog import ensure_catalog_table, load_schema
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema, pool_table_names
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project import project_service as project_service_module


def _get_dsn() -> str:
    dsn = os.getenv("DR_LLM_TEST_DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "Set DR_LLM_TEST_DATABASE_URL to run pool migration integration tests"
        )
    return dsn


def _project_for_dsn(name: str, dsn: str) -> ProjectInfo:
    parsed = urlparse(dsn)
    assert parsed.port is not None
    return ProjectInfo(name=name, port=parsed.port, status=ContainerStatus.RUNNING)


def _drop_pool_artifacts(dsn: str, pool_name: str) -> None:
    """Drop the pool's tables and clean up ``pool_catalog`` for cross-test safety.

    Drops the per-pool tables, removes the pool's row from ``pool_catalog`` if
    that table exists, and re-creates ``pool_catalog`` if a test simulated a
    pre-migration project by dropping it. This keeps the shared Postgres
    instance in a consistent state for tests that run after these.
    """
    with psycopg.connect(dsn) as conn:
        for table_name in pool_table_names(pool_name):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", table_name)
                )
            )
        catalog_exists = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='pool_catalog')"
        ).fetchone()
        if catalog_exists is not None and catalog_exists[0]:
            conn.execute("DELETE FROM pool_catalog WHERE pool_name = %s", [pool_name])
        conn.commit()
    # Restore pool_catalog if a test simulated pre-migration state.
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    try:
        ensure_catalog_table(runtime)
    finally:
        runtime.close()


def _seed_pool_with_samples(
    dsn: str, schema: PoolSchema, *, samples: int = 1
) -> PoolStore:
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    store = PoolStore(schema, runtime)
    try:
        store.ensure_schema()
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        runtime.close()
        pytest.skip(f"Postgres unavailable for pool migration tests: {exc}")
    for i in range(samples):
        store.insert_sample(
            PoolSample(
                key_values={kc.name: f"v{i}" for kc in schema.key_columns},
                sample_idx=i,
                request={"prompt": "hello"},
            )
        )
    return store


def _delete_catalog_table(dsn: str) -> None:
    """Drop pool_catalog to simulate a pre-migration project."""
    with psycopg.connect(dsn) as conn:
        conn.execute("DROP TABLE IF EXISTS pool_catalog CASCADE")
        conn.commit()


@pytest.mark.integration
def test_derive_pool_schema_recovers_text_keys() -> None:
    dsn = _get_dsn()
    pool_name = f"mig_{uuid4().hex[:8]}"
    schema = PoolSchema(
        name=pool_name,
        key_columns=[KeyColumn(name="model"), KeyColumn(name="prompt")],
    )
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    try:
        store = _seed_pool_with_samples(dsn, schema)
        derived = derive_pool_schema(runtime, pool_name)
        assert derived.name == pool_name
        assert [kc.name for kc in derived.key_columns] == ["model", "prompt"]
        assert all(kc.type == ColumnType.text for kc in derived.key_columns)
        store._runtime.close()
    finally:
        runtime.close()
        _drop_pool_artifacts(dsn, pool_name)


@pytest.mark.integration
def test_derive_pool_schema_recovers_mixed_types() -> None:
    dsn = _get_dsn()
    pool_name = f"mig_{uuid4().hex[:8]}"
    schema = PoolSchema(
        name=pool_name,
        key_columns=[
            KeyColumn(name="model", type=ColumnType.text),
            KeyColumn(name="seed", type=ColumnType.integer),
            KeyColumn(name="enabled", type=ColumnType.boolean),
            KeyColumn(name="weight", type=ColumnType.float_),
        ],
    )
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    try:
        store = _seed_pool_with_samples(dsn, schema, samples=0)
        derived = derive_pool_schema(runtime, pool_name)
        derived_by_name = {kc.name: kc.type for kc in derived.key_columns}
        assert derived_by_name == {
            "model": ColumnType.text,
            "seed": ColumnType.integer,
            "enabled": ColumnType.boolean,
            "weight": ColumnType.float_,
        }
        store._runtime.close()
    finally:
        runtime.close()
        _drop_pool_artifacts(dsn, pool_name)


@pytest.mark.integration
def test_derive_pool_schema_raises_when_table_missing() -> None:
    dsn = _get_dsn()
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
    try:
        with pytest.raises(PoolMigrationError):
            derive_pool_schema(runtime, f"nonexistent_{uuid4().hex[:8]}")
    finally:
        runtime.close()


@pytest.mark.integration
def test_backfill_project_catalog_writes_catalog_for_pre_migration_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    pool_name = f"mig_{uuid4().hex[:8]}"
    schema = PoolSchema(
        name=pool_name,
        key_columns=[KeyColumn(name="model"), KeyColumn(name="prompt")],
    )

    try:
        _seed_pool_with_samples(dsn, schema)
        # Simulate a project that predates pool_catalog: drop the table entirely.
        _delete_catalog_table(dsn)

        monkeypatch.setattr(
            project_service_module,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = backfill_project_catalog(
            BackfillProjectCatalogRequest(project_name="demo")
        )

        assert result.success
        assert pool_name in result.discovered_pool_names
        pool_results = {r.pool_name: r for r in result.pool_results}
        assert pool_results[pool_name].status == PoolBackfillStatus.backfilled

        runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=2))
        try:
            persisted = load_schema(runtime, pool_name)
        finally:
            runtime.close()
        assert persisted is not None
        assert [kc.name for kc in persisted.key_columns] == ["model", "prompt"]
    finally:
        _drop_pool_artifacts(dsn, pool_name)


@pytest.mark.integration
def test_backfill_project_catalog_skips_already_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    pool_name = f"mig_{uuid4().hex[:8]}"
    schema = PoolSchema(name=pool_name, key_columns=[KeyColumn(name="model")])

    try:
        _seed_pool_with_samples(dsn, schema)
        # ensure_schema in the seed already wrote a catalog entry; backfill should skip.
        monkeypatch.setattr(
            project_service_module,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = backfill_project_catalog(
            BackfillProjectCatalogRequest(project_name="demo")
        )

        assert result.success
        pool_results = {r.pool_name: r for r in result.pool_results}
        assert pool_results[pool_name].status == PoolBackfillStatus.already_persisted
    finally:
        _drop_pool_artifacts(dsn, pool_name)


@pytest.mark.integration
def test_backfill_project_catalog_dry_run_reports_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = _get_dsn()
    pool_name = f"mig_{uuid4().hex[:8]}"
    schema = PoolSchema(name=pool_name, key_columns=[KeyColumn(name="model")])

    try:
        _seed_pool_with_samples(dsn, schema)
        _delete_catalog_table(dsn)

        monkeypatch.setattr(
            project_service_module,
            "maybe_get_project",
            lambda name: _project_for_dsn(name, dsn),
        )

        result = backfill_project_catalog(
            BackfillProjectCatalogRequest(project_name="demo", dry_run=True)
        )

        assert result.success
        pool_results = {r.pool_name: r for r in result.pool_results}
        assert pool_results[pool_name].status == PoolBackfillStatus.would_backfill

        # Catalog table should not have been created by dry run.
        with psycopg.connect(dsn) as conn:
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='pool_catalog')"
            ).fetchone()
        assert row is not None and row[0] is False
    finally:
        _drop_pool_artifacts(dsn, pool_name)
