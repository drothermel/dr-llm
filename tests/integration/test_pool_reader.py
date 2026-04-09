"""Integration tests for PoolReader (requires PostgreSQL)."""

from __future__ import annotations

import os
from collections.abc import Generator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import text

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.errors import PoolNotFoundError, PoolSchemaNotPersistedError
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus
from dr_llm.pool.pool_sample import PoolSample, SampleStatus
from dr_llm.pool.pool_store import SCHEMA_METADATA_KEY, PoolStore
from dr_llm.pool.reader import PoolReader

_READER_SCHEMA = PoolSchema(
    name="itest_reader",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)

_READER_TABLES = (
    _READER_SCHEMA.metadata_table,
    _READER_SCHEMA.claims_table,
    _READER_SCHEMA.pending_table,
    _READER_SCHEMA.samples_table,
)


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _drop_pool_tables(dsn: str, schema: PoolSchema) -> None:
    with psycopg.connect(dsn) as conn:
        for tbl in (
            schema.metadata_table,
            schema.claims_table,
            schema.pending_table,
            schema.samples_table,
        ):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", tbl)
                )
            )
        conn.commit()


@pytest.fixture(scope="module")
def reader_runtime() -> Generator[DbRuntime, None, None]:
    """Module-scoped DbRuntime for reader integration tests."""
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")

    runtime: DbRuntime | None = None
    try:
        _drop_pool_tables(dsn, _READER_SCHEMA)
        runtime = DbRuntime(
            DbConfig(
                dsn=dsn,
                min_pool_size=1,
                max_pool_size=4,
                application_name="pool_reader_tests",
            )
        )
        # Seed shared pool with a few samples + pending rows of varied statuses.
        store = PoolStore(_READER_SCHEMA, runtime)
        store.ensure_schema()
        for i in range(3):
            store.insert_sample(
                PoolSample(
                    key_values={"dim_a": "alpha", "dim_b": 1},
                    sample_idx=i,
                    payload={"i": i},
                    metadata={"seeded": True},
                )
            )
        for i in range(2):
            store.insert_sample(
                PoolSample(
                    key_values={"dim_a": "beta", "dim_b": 2},
                    sample_idx=i,
                    payload={"i": i},
                )
            )
        # Pending rows: one pending, one will be promoted, one will be failed.
        store.pending.insert(
            PendingSample(
                key_values={"dim_a": "alpha", "dim_b": 1},
                sample_idx=10,
                payload={"draft": "p1"},
            )
        )
        promoted_pending = PendingSample(
            key_values={"dim_a": "alpha", "dim_b": 1},
            sample_idx=11,
            payload={"draft": "p2"},
        )
        store.pending.insert(promoted_pending)
        failed_pending = PendingSample(
            key_values={"dim_a": "beta", "dim_b": 2},
            sample_idx=20,
            payload={"draft": "p3"},
        )
        store.pending.insert(failed_pending)

        # Promote the second pending row, fail the third.
        claimed_promote = store.pending.claim(
            worker_id="w-promote",
            lease_seconds=60,
            key_filter={"dim_a": "alpha", "dim_b": 1},
        )
        assert claimed_promote is not None
        # claim() returns oldest first, so we may have grabbed pending #1 not #2;
        # promote whichever we got and claim again to promote the other.
        store.pending.promote(
            pending_id=claimed_promote.pending_id,
            worker_id="w-promote",
            payload={"final": True},
        )

        claimed_fail = store.pending.claim(
            worker_id="w-fail",
            lease_seconds=60,
            key_filter={"dim_a": "beta", "dim_b": 2},
        )
        assert claimed_fail is not None
        store.pending.fail(
            pending_id=claimed_fail.pending_id, worker_id="w-fail", reason="test"
        )

        # Seed metadata for prefix-scan tests.
        store.metadata.upsert("prompt_template/foo", {"template_text": "hello {{X}}"})
        store.metadata.upsert("prompt_template/bar", {"template_text": "world"})
        store.metadata.upsert("data_sample/baz", {"source_code": "def f(): ..."})

    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        if runtime is not None:
            runtime.close()
        pytest.skip(f"Postgres unavailable for pool reader integration tests: {exc}")

    yield runtime
    _drop_pool_tables(dsn, _READER_SCHEMA)
    runtime.close()


@pytest.mark.integration
def test_from_runtime_happy_path(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    assert reader.pool_name == _READER_SCHEMA.name
    assert reader.schema == _READER_SCHEMA


@pytest.mark.integration
def test_samples_list_returns_all_seeded(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    samples = reader.samples_list()
    # 3 alpha + 2 beta + 1 promoted alpha = 6
    assert len(samples) == 6


@pytest.mark.integration
def test_samples_list_with_key_filter(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    alpha = reader.samples_list(key_filter={"dim_a": "alpha"})
    assert all(s.key_values["dim_a"] == "alpha" for s in alpha)
    # 3 originally seeded + 1 promoted from pending
    assert len(alpha) == 4


@pytest.mark.integration
def test_samples_list_with_status_filter(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    active = reader.samples_list(status=SampleStatus.active)
    assert all(s.status == SampleStatus.active for s in active)
    assert len(active) == 6  # all seeded samples are active


@pytest.mark.integration
def test_samples_streaming_iterator(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    streamed = list(reader.samples(key_filter={"dim_a": "beta"}))
    assert len(streamed) == 2
    assert all(s.key_values["dim_a"] == "beta" for s in streamed)


@pytest.mark.integration
def test_pending_default_returns_in_flight_only(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    in_flight = reader.pending_list()
    # 3 inserted, 1 promoted, 1 failed → 1 still in-flight (pending status)
    assert len(in_flight) == 1
    assert in_flight[0].status in {PendingStatus.pending, PendingStatus.leased}


@pytest.mark.integration
def test_pending_status_filter_includes_terminal(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    promoted = reader.pending_list(status=PendingStatus.promoted)
    assert len(promoted) == 1
    assert promoted[0].status == PendingStatus.promoted

    failed = reader.pending_list(status=PendingStatus.failed)
    assert len(failed) == 1
    assert failed[0].status == PendingStatus.failed

    all_states = reader.pending_list(
        status=[
            PendingStatus.pending,
            PendingStatus.leased,
            PendingStatus.promoted,
            PendingStatus.failed,
        ]
    )
    assert len(all_states) == 3


@pytest.mark.integration
def test_progress_aggregates_correctly(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    progress = reader.progress()
    assert progress.samples_total == 6
    assert progress.pending_counts.promoted == 1
    assert progress.pending_counts.failed == 1
    # 3 inserted - 1 promoted - 1 failed = 1 still pending
    assert progress.pending_counts.pending + progress.pending_counts.leased == 1
    assert progress.in_flight == 1
    assert progress.is_complete is False


@pytest.mark.integration
def test_metadata_get_returns_value(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    value = reader.metadata_get("prompt_template/foo")
    assert value == {"template_text": "hello {{X}}"}


@pytest.mark.integration
def test_metadata_get_returns_none_for_missing(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    assert reader.metadata_get("nonexistent_key") is None


@pytest.mark.integration
def test_metadata_prefix_scans_subset(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    templates = reader.metadata_prefix("prompt_template/")
    assert set(templates.keys()) == {"prompt_template/foo", "prompt_template/bar"}
    assert templates["prompt_template/foo"]["template_text"] == "hello {{X}}"

    samples = reader.metadata_prefix("data_sample/")
    assert set(samples.keys()) == {"data_sample/baz"}


@pytest.mark.integration
def test_metadata_prefix_can_load_internal_schema_key(
    reader_runtime: DbRuntime,
) -> None:
    """The persisted schema should be discoverable via prefix scan."""
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    underscore_keys = reader.metadata_prefix("_")
    assert SCHEMA_METADATA_KEY in underscore_keys
    # Round-trip: the persisted dump should validate back into a PoolSchema.
    reconstructed = PoolSchema.model_validate(underscore_keys[SCHEMA_METADATA_KEY])
    assert reconstructed == _READER_SCHEMA


@pytest.mark.integration
def test_ensure_schema_persists_schema_metadata(reader_runtime: DbRuntime) -> None:
    """ensure_schema must populate the _schema metadata key."""
    store = PoolStore(_READER_SCHEMA, reader_runtime)
    persisted = store.metadata.get(SCHEMA_METADATA_KEY)
    assert persisted is not None
    reconstructed = PoolSchema.model_validate(persisted)
    assert reconstructed == _READER_SCHEMA


# --- Schema-loading classmethod paths ---


def _isolated_pool_schema() -> PoolSchema:
    return PoolSchema(
        name=f"itest_reader_iso_{uuid4().hex[:8]}",
        key_columns=[KeyColumn(name="dim_a")],
    )


@pytest.mark.integration
def test_load_schema_from_db_round_trip() -> None:
    """A reader constructed without an explicit schema should reconstruct
    the original PoolSchema from the metadata table."""
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    schema = _isolated_pool_schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_iso",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()

        from dr_llm.pool.reader import _load_schema_from_db

        loaded = _load_schema_from_db(runtime, schema.name)
        assert loaded == schema
    finally:
        _drop_pool_tables(dsn, schema)
        runtime.close()


@pytest.mark.integration
def test_load_schema_from_db_raises_when_pool_missing() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_missing",
        )
    )
    try:
        from dr_llm.pool.reader import _load_schema_from_db

        with pytest.raises(PoolNotFoundError):
            _load_schema_from_db(runtime, f"never_created_{uuid4().hex[:8]}")
    finally:
        runtime.close()


@pytest.mark.integration
def test_load_schema_from_db_raises_when_schema_row_missing() -> None:
    """Migration path: pool exists but _schema row is gone (pre-feature pool)."""
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    schema = _isolated_pool_schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_migration",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()

        # Simulate a pool created before the persist-schema feature.
        with psycopg.connect(dsn) as conn:
            conn.execute(
                sql.SQL("DELETE FROM {} WHERE key = %s").format(
                    sql.Identifier("public", schema.metadata_table)
                ),
                (SCHEMA_METADATA_KEY,),
            )
            conn.commit()

        from dr_llm.pool.reader import _load_schema_from_db

        with pytest.raises(PoolSchemaNotPersistedError):
            _load_schema_from_db(runtime, schema.name)

        # Recovery: from_runtime with explicit schema works without backfilling.
        reader = PoolReader.from_runtime(runtime, schema=schema)
        assert reader.schema == schema

        # Re-running ensure_schema backfills the _schema row.
        store.ensure_schema()
        assert _load_schema_from_db(runtime, schema.name) == schema
    finally:
        _drop_pool_tables(dsn, schema)
        runtime.close()


@pytest.mark.integration
def test_close_disposes_owned_runtime_only() -> None:
    """from_runtime borrows; close() must NOT dispose a borrowed runtime."""
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    schema = _isolated_pool_schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_lifecycle",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()

        reader = PoolReader.from_runtime(runtime, schema=schema)
        reader.close()
        # close() is idempotent and the borrowed runtime is still alive.
        reader.close()
        # The runtime is still usable after the borrowed reader closes.
        with runtime.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        _drop_pool_tables(dsn, schema)
        runtime.close()


@pytest.mark.integration
def test_context_manager_calls_close() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    schema = _isolated_pool_schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_ctx",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()

        with PoolReader.from_runtime(runtime, schema=schema) as reader:
            assert reader.samples_list() == []
        # The borrowed runtime survives context exit.
        with runtime.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        _drop_pool_tables(dsn, schema)
        runtime.close()
