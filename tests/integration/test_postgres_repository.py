from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_pool.errors import SessionConflictError
from llm_pool.storage.repository import PostgresRepository, StorageConfig
from llm_pool.types import SessionTurnStatus, ToolPolicy


@pytest.fixture(scope="module")
def repository() -> PostgresRepository:
    dsn = os.getenv("LLM_POOL_TEST_DATABASE_URL") or os.getenv("LLM_POOL_DATABASE_URL")
    if not dsn:
        pytest.skip("Set LLM_POOL_TEST_DATABASE_URL (or LLM_POOL_DATABASE_URL) to run integration tests")
    repo = PostgresRepository(
        StorageConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=16,
            application_name="llm_pool_tests",
        )
    )
    repo.init_schema()
    yield repo
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
    turn_id, _ = repository.create_session_turn(session_id=handle.session_id, status=SessionTurnStatus.active)
    repository.append_session_event(
        session_id=handle.session_id,
        turn_id=turn_id,
        event_type="message",
        payload={"message": {"role": "user", "content": "hello"}},
    )
    repository.complete_session_turn(turn_id=turn_id, status=SessionTurnStatus.completed)

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
        repository.advance_session_version(session_id=handle.session_id, expected_version=1)


@pytest.mark.integration
def test_tool_claim_parallel_without_duplicates(repository: PostgresRepository) -> None:
    handle = repository.start_session(
        strategy_mode=ToolPolicy.native_preferred,
        metadata={"provider": "openai", "model": "gpt-4.1"},
    )
    turn_id, _ = repository.create_session_turn(session_id=handle.session_id, status=SessionTurnStatus.active)

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
            batch = repository.claim_tool_calls(worker_id=worker, limit=3, lease_seconds=120)
            if not batch:
                return claimed_ids
            claimed_ids.extend(call.tool_call_id for call in batch)

    workers = [f"w{idx}" for idx in range(8)]
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        results = [future.result() for future in [pool.submit(claim_all, worker) for worker in workers]]

    flattened = [item for sublist in results for item in sublist]
    assert len(flattened) == 24
    assert len(set(flattened)) == 24
