from __future__ import annotations

import os
from collections.abc import Generator

import psycopg
import pytest
from psycopg import sql

from dr_llm.pool.db import PoolDb
from dr_llm.pool.runtime import DbConfig
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode, Message
from dr_llm.providers.usage import TokenUsage

_TEST_TABLES = (
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
def db() -> Generator[PoolDb, None, None]:
    dsn = os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")
    if not dsn:
        pytest.skip(
            "Set DR_LLM_TEST_DATABASE_URL (or DR_LLM_DATABASE_URL) to run integration tests"
        )
    pool_db = PoolDb(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=16,
            application_name="dr_llm_tests",
        )
    )
    pool_db.init_schema()
    _truncate_test_tables(dsn)
    yield pool_db
    _truncate_test_tables(dsn)
    pool_db.close()


@pytest.mark.integration
def test_schema_bootstrap_idempotent(db: PoolDb) -> None:
    db.init_schema()
    db.init_schema()


@pytest.mark.integration
def test_schema_migration_removes_legacy_tables(db: PoolDb) -> None:
    assert db.config is not None
    dsn = db.config.dsn
    assert dsn is not None
    legacy_table_names = [
        "sessions",
        "session_turns",
        "session_events",
        "tool_calls",
        "tool_results",
        "tool_call_dead_letters",
    ]
    with psycopg.connect(dsn) as conn:
        for table_name in legacy_table_names:
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY
                    )
                    """
                ).format(sql.Identifier(table_name))
            )
        conn.commit()

    migrated_db = PoolDb(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=16,
            application_name="dr_llm_migration_tests",
        )
    )
    try:
        migrated_db.init_schema()
    finally:
        migrated_db.close()

    with psycopg.connect(dsn) as conn:
        for table_name in legacy_table_names:
            table_exists = conn.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                ) AS exists
                """,
                (table_name,),
            ).fetchone()
            assert table_exists is not None
            assert table_exists[0] is False


@pytest.mark.integration
def test_record_call_and_list_calls_round_trip(db: PoolDb) -> None:
    call_id = db.record_call(
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

    calls = db.list_calls(limit=10)
    matching_call = next((call for call in calls if call.call_id == call_id), None)
    assert matching_call is not None
    assert matching_call.provider == "openai"
    assert matching_call.model == "gpt-4.1"
    assert matching_call.request is not None
    assert matching_call.request["metadata"]["purpose"] == "test"
    assert matching_call.response is not None
    assert matching_call.response["usage"]["total_tokens"] == 3
