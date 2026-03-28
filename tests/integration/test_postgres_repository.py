from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any, cast

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from dr_llm.storage.repository import PostgresRepository, StorageConfig
from dr_llm.types import CallMode, LlmRequest, LlmResponse, Message, TokenUsage

_TEST_TABLES = (
    "provider_models_current",
    "provider_model_catalog_snapshots",
    "artifacts",
    "llm_call_responses",
    "llm_call_requests",
    "llm_calls",
    "run_parameters",
    "runs",
)


def _truncate_test_tables(dsn: str) -> None:
    table_list = sql.SQL(", ").join(sql.Identifier(name) for name in _TEST_TABLES)
    stmt = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(table_list)
    with psycopg.connect(dsn) as conn:
        conn.execute(stmt)
        conn.commit()


@pytest.fixture(scope="module")
def repository() -> Generator[PostgresRepository, None, None]:
    dsn = os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")
    if not dsn:
        pytest.skip(
            "Set DR_LLM_TEST_DATABASE_URL (or DR_LLM_DATABASE_URL) to run integration tests"
        )
    repo = PostgresRepository(
        StorageConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=16,
            application_name="dr_llm_tests",
        )
    )
    repo.init_schema()
    _truncate_test_tables(dsn)
    yield repo
    _truncate_test_tables(dsn)
    repo.close()


@pytest.mark.integration
def test_schema_bootstrap_idempotent(repository: PostgresRepository) -> None:
    repository.init_schema()
    repository.init_schema()


@pytest.mark.integration
def test_schema_migration_removes_legacy_tables_and_supports_tools_column(
    repository: PostgresRepository,
) -> None:
    dsn = repository.config.dsn
    with psycopg.connect(dsn) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                tool_call_id TEXT PRIMARY KEY
            )
            """
        )
        conn.execute(
            """
            ALTER TABLE provider_models_current
            ADD COLUMN IF NOT EXISTS supports_tools BOOLEAN
            """
        )
        conn.commit()

    migrated_repository = PostgresRepository(
        StorageConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=16,
            application_name="dr_llm_migration_tests",
        )
    )
    try:
        migrated_repository.init_schema()
    finally:
        migrated_repository.close()

    with psycopg.connect(dsn) as conn:
        sessions_exists = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'sessions'
            ) AS exists
            """
        ).fetchone()
        assert sessions_exists is not None
        assert sessions_exists[0] is False

        tool_calls_exists = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'tool_calls'
            ) AS exists
            """
        ).fetchone()
        assert tool_calls_exists is not None
        assert tool_calls_exists[0] is False

        supports_tools_exists = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'provider_models_current'
                  AND column_name = 'supports_tools'
            ) AS exists
            """
        ).fetchone()
        assert supports_tools_exists is not None
        assert supports_tools_exists[0] is False


@pytest.mark.integration
def test_record_call_and_list_calls_round_trip(repository: PostgresRepository) -> None:
    call_id = repository.record_call(
        request=LlmRequest(
            provider="openai",
            model="gpt-4.1",
            messages=[Message(role="user", content="hello")],
            metadata={"purpose": "test"},
        ),
        response=LlmResponse(
            text="hi",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            provider="openai",
            model="gpt-4.1",
            mode=CallMode.api,
        ),
        status="success",
        mode=CallMode.api,
        metadata={"source": "integration"},
    )

    calls = repository.list_calls(limit=10)
    assert any(call.call_id == call_id for call in calls)
