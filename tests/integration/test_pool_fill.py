"""Integration tests for pool fill helpers (requires PostgreSQL)."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db import DbConfig, DbRuntime, KeyColumn, PoolSchema
from dr_llm.pool.pending.pool_worker_controller import PoolWorkerController
from dr_llm.pool.pending.workers import seed_pending, start_workers
from dr_llm.pool.sample_store import PoolStore


_TEST_SCHEMA = PoolSchema(
    name="itest_fill",
    key_columns=[KeyColumn(name="model"), KeyColumn(name="prompt")],
)

_POOL_TABLES = (
    _TEST_SCHEMA.metadata_table,
    _TEST_SCHEMA.claims_table,
    _TEST_SCHEMA.pending_table,
    _TEST_SCHEMA.samples_table,
)


def _drop_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for table_name in _POOL_TABLES:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(table_name)
                )
            )
        conn.commit()


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


@pytest.fixture
def fill_store() -> Generator[PoolStore, None, None]:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    runtime: DbRuntime | None = None
    try:
        _drop_tables(dsn)
        runtime = DbRuntime(
            DbConfig(
                dsn=dsn,
                min_pool_size=1,
                max_pool_size=4,
                application_name="pool_fill_tests",
            )
        )
        store = PoolStore(_TEST_SCHEMA, runtime)
        store.init_schema()
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        if runtime is not None:
            runtime.close()
        pytest.skip(f"Postgres unavailable for pool fill integration tests: {exc}")
    yield store
    _drop_tables(dsn)
    runtime.close()


def _wait_for_terminal_queue(
    store: PoolStore,
    *,
    key_filter: dict[str, str] | None = None,
    timeout_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        counts = store.pending.status_counts(key_filter=key_filter)
        if counts.pending == 0 and counts.leased == 0:
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for queue to reach a terminal state")


def _stop_controller(controller: PoolWorkerController) -> None:
    controller.stop()
    controller.join(timeout=5.0)


@pytest.mark.integration
def test_pending_status_counts_defaults(fill_store: PoolStore) -> None:
    counts = fill_store.pending.status_counts()

    assert counts.pending == 0
    assert counts.leased == 0
    assert counts.promoted == 0
    assert counts.failed == 0
    assert counts.total == 0


@pytest.mark.integration
def test_start_workers_promote_seeded_grid(fill_store: PoolStore) -> None:
    seed_result = seed_pending(
        fill_store,
        key_grid={
            "model": ["gpt-4.1-mini", "gpt-5-mini"],
            "prompt": ["math", "history"],
        },
        n=2,
    )
    assert seed_result.inserted == 8

    controller = start_workers(
        fill_store,
        process_fn=lambda sample: {
            "completion": (
                f"{sample.key_values['model']}::{sample.key_values['prompt']}::"
                f"{sample.sample_idx}"
            )
        },
        num_workers=3,
        min_poll_interval_s=0.01,
        max_poll_interval_s=0.05,
    )
    try:
        _wait_for_terminal_queue(fill_store)
        controller.stop()
        snapshot = controller.join(timeout=5.0)
    finally:
        _stop_controller(controller)

    assert snapshot.counts.claimed == 8
    assert snapshot.counts.promoted == 8
    assert snapshot.counts.failed == 0
    assert snapshot.status_counts.pending == 0
    assert snapshot.status_counts.leased == 0
    assert snapshot.status_counts.promoted == 8
    for model in ["gpt-4.1-mini", "gpt-5-mini"]:
        for prompt in ["math", "history"]:
            assert fill_store.cell_depth(key_values={"model": model, "prompt": prompt}) == 2


@pytest.mark.integration
def test_start_workers_retry_then_fail(fill_store: PoolStore) -> None:
    seed_pending(
        fill_store,
        key_grid={"model": ["retry-model"], "prompt": ["fail-prompt"]},
        n=1,
    )

    def always_fail(_sample: object) -> dict[str, str]:
        raise RuntimeError("boom")

    controller = start_workers(
        fill_store,
        process_fn=always_fail,
        num_workers=1,
        min_poll_interval_s=0.01,
        max_poll_interval_s=0.05,
        max_retries=1,
    )
    try:
        _wait_for_terminal_queue(fill_store)
        controller.stop()
        snapshot = controller.join(timeout=5.0)
    finally:
        _stop_controller(controller)

    assert snapshot.counts.claimed == 2
    assert snapshot.counts.retried == 1
    assert snapshot.counts.failed == 1
    assert snapshot.counts.process_errors == 2
    assert snapshot.status_counts.failed == 1
    assert fill_store.cell_depth(
        key_values={"model": "retry-model", "prompt": "fail-prompt"}
    ) == 0


@pytest.mark.integration
def test_start_workers_retry_then_promote(fill_store: PoolStore) -> None:
    seed_pending(
        fill_store,
        key_grid={"model": ["retry-model"], "prompt": ["eventual-success"]},
        n=1,
    )
    attempts = {"count": 0}

    def flaky_process(_sample: object) -> dict[str, str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary error")
        return {"completion": "ok"}

    controller = start_workers(
        fill_store,
        process_fn=flaky_process,
        num_workers=1,
        min_poll_interval_s=0.01,
        max_poll_interval_s=0.05,
        max_retries=1,
    )
    try:
        _wait_for_terminal_queue(fill_store)
        controller.stop()
        snapshot = controller.join(timeout=5.0)
    finally:
        _stop_controller(controller)

    assert snapshot.counts.claimed == 2
    assert snapshot.counts.retried == 1
    assert snapshot.counts.failed == 0
    assert snapshot.counts.promoted == 1
    assert snapshot.counts.process_errors == 1
    assert snapshot.status_counts.promoted == 1
    assert fill_store.cell_depth(
        key_values={"model": "retry-model", "prompt": "eventual-success"}
    ) == 1


@pytest.mark.integration
def test_start_workers_respects_key_filter(fill_store: PoolStore) -> None:
    seed_pending(
        fill_store,
        key_grid={"model": ["m1", "m2"], "prompt": ["shared"]},
        n=2,
    )

    controller = start_workers(
        fill_store,
        process_fn=lambda sample: {"completion": sample.key_values["model"]},
        num_workers=2,
        min_poll_interval_s=0.01,
        max_poll_interval_s=0.05,
        key_filter={"model": "m1"},
    )
    try:
        _wait_for_terminal_queue(fill_store, key_filter={"model": "m1"})
        controller.stop()
        snapshot = controller.join(timeout=5.0)
    finally:
        _stop_controller(controller)
    all_counts = fill_store.pending.status_counts()

    assert snapshot.counts.promoted == 2
    assert snapshot.status_counts.pending == 0
    assert snapshot.status_counts.promoted == 2
    assert all_counts.pending == 2
    assert all_counts.promoted == 2
    assert fill_store.cell_depth(key_values={"model": "m1", "prompt": "shared"}) == 2
    assert fill_store.cell_depth(key_values={"model": "m2", "prompt": "shared"}) == 0


@pytest.mark.integration
def test_seed_pending_rich_grid_with_workers(fill_store: PoolStore) -> None:
    """End-to-end: seed with rich grid values, fill with make_llm_process_fn."""
    from unittest.mock import MagicMock

    from dr_llm.pool.pending.workers import make_llm_process_fn, seed_pending
    from dr_llm.providers.llm_config import LlmConfig
    from dr_llm.providers.llm_response import LlmResponse
    from dr_llm.providers.models import CallMode, Message
    from dr_llm.providers.usage import TokenUsage

    # Use a fresh schema with llm_config/prompt columns
    fill_schema = PoolSchema(
        name=f"itest_rich_{uuid4().hex[:8]}",
        key_columns=[KeyColumn(name="llm_config"), KeyColumn(name="prompt")],
    )
    rich_store = PoolStore(fill_schema, fill_store._runtime)
    rich_store.init_schema()

    try:
        configs = {
            "cfg_a": LlmConfig(provider="fake", model="fake-model"),
        }
        prompts = {
            "p1": [Message(role="user", content="hello")],
        }
        seed_result = seed_pending(
            rich_store,
            key_grid={"llm_config": configs, "prompt": prompts},
            n=2,
        )
        assert seed_result.inserted == 2

        fake_response = LlmResponse(
            text="fake response",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider="fake",
            model="fake-model",
            mode=CallMode.api,
        )
        adapter = MagicMock()
        adapter.generate.return_value = fake_response
        registry = MagicMock()
        registry.get.return_value = adapter

        process_fn = make_llm_process_fn(registry)
        controller = start_workers(
            rich_store,
            process_fn=process_fn,
            num_workers=2,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        )
        try:
            _wait_for_terminal_queue(rich_store)
            controller.stop()
            snapshot = controller.join(timeout=5.0)
        finally:
            _stop_controller(controller)

        assert snapshot.counts.promoted == 2
        assert snapshot.counts.failed == 0
        samples = rich_store.bulk_load()
        assert len(samples) == 2
        assert all(s.payload["text"] == "fake response" for s in samples)
    finally:
        dsn = _get_dsn()
        if dsn:
            rich_tables = (
                fill_schema.samples_table,
                fill_schema.claims_table,
                fill_schema.pending_table,
                fill_schema.metadata_table,
            )
            with psycopg.connect(dsn) as conn:
                for table_name in rich_tables:
                    conn.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(table_name)
                        )
                    )
                conn.commit()
