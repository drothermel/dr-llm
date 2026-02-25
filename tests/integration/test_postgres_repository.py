from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Generator
from typing import Any, cast

import pytest
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from llm_pool.benchmark import BenchmarkConfig, run_repository_benchmark
from llm_pool.errors import SessionConflictError
from llm_pool.storage.repository import PostgresRepository, StorageConfig
from llm_pool.types import SessionTurnStatus, ToolPolicy

_TEST_TABLES = (
    "tool_call_dead_letters",
    "tool_results",
    "tool_calls",
    "provider_model_overrides",
    "provider_models_current",
    "provider_model_catalog_snapshots",
    "session_events",
    "session_turns",
    "sessions",
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
    dsn = os.getenv("LLM_POOL_TEST_DATABASE_URL") or os.getenv("LLM_POOL_DATABASE_URL")
    if not dsn:
        pytest.skip(
            "Set LLM_POOL_TEST_DATABASE_URL (or LLM_POOL_DATABASE_URL) to run integration tests"
        )
    repo = PostgresRepository(
        StorageConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=16,
            application_name="llm_pool_tests",
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
def test_session_event_append_and_replay(repository: PostgresRepository) -> None:
    handle = repository.start_session(
        strategy_mode=ToolPolicy.native_preferred,
        metadata={"provider": "openai", "model": "gpt-4.1"},
    )
    turn_id, _ = repository.create_session_turn(
        session_id=handle.session_id, status=SessionTurnStatus.active
    )
    repository.append_session_event(
        session_id=handle.session_id,
        turn_id=turn_id,
        event_type="message",
        payload={"message": {"role": "user", "content": "hello"}},
    )
    repository.complete_session_turn(
        turn_id=turn_id, status=SessionTurnStatus.completed
    )

    replayed = repository.replay_session_messages(session_id=handle.session_id)
    assert replayed[-1] == {"role": "user", "content": "hello"}


@pytest.mark.integration
def test_session_version_conflict(repository: PostgresRepository) -> None:
    handle = repository.start_session(
        strategy_mode=ToolPolicy.native_preferred,
        metadata={"provider": "openai", "model": "gpt-4.1"},
    )
    repository.advance_session_version(session_id=handle.session_id, expected_version=1)
    with pytest.raises(SessionConflictError):
        repository.advance_session_version(
            session_id=handle.session_id, expected_version=1
        )


@pytest.mark.integration
def test_tool_claim_parallel_without_duplicates(repository: PostgresRepository) -> None:
    handle = repository.start_session(
        strategy_mode=ToolPolicy.native_preferred,
        metadata={"provider": "openai", "model": "gpt-4.1"},
    )
    turn_id, _ = repository.create_session_turn(
        session_id=handle.session_id, status=SessionTurnStatus.active
    )

    for idx in range(24):
        repository.enqueue_tool_call(
            session_id=handle.session_id,
            turn_id=turn_id,
            tool_name="echo",
            args={"idx": idx},
            idempotency_key=f"{handle.session_id}:{turn_id}:{idx}",
            tool_call_id=f"tc_{idx}",
        )

    def claim_all(worker: str) -> list[str]:
        claimed_ids: list[str] = []
        while True:
            batch = repository.claim_tool_calls(
                worker_id=worker, limit=3, lease_seconds=120
            )
            if not batch:
                return claimed_ids
            claimed_ids.extend(call.tool_call_id for call in batch)

    workers = [f"w{idx}" for idx in range(8)]
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        results = [
            future.result()
            for future in [pool.submit(claim_all, worker) for worker in workers]
        ]

    flattened = [item for sublist in results for item in sublist]
    assert len(flattened) == 24
    assert len(set(flattened)) == 24


@pytest.mark.integration
def test_benchmark_persists_artifact_record(
    repository: PostgresRepository, tmp_path
) -> None:
    artifact_path = tmp_path / "benchmark-report.json"

    report = run_repository_benchmark(
        repository=repository,
        config=BenchmarkConfig(
            workers=4,
            total_operations=120,
            warmup_operations=12,
            max_in_flight=4,
            artifact_path=str(artifact_path),
        ),
    )

    assert report.run_id
    assert artifact_path.exists()
    with psycopg.connect(
        repository.config.dsn, row_factory=cast(Any, dict_row)
    ) as conn:
        row = conn.execute(
            """
            SELECT artifact_type, artifact_path
            FROM artifacts
            WHERE run_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [report.run_id],
        ).fetchone()
    assert row is not None
    row_dict = cast(dict[str, Any], row)
    assert row_dict["artifact_type"] == "benchmark_report"
    assert str(row_dict["artifact_path"]) == str(artifact_path)
