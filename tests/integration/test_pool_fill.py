"""Integration tests for pool fill helpers (requires PostgreSQL)."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.pool_fill import seed_pending, start_workers
from dr_llm.pool.pool_schema import KeyColumn, PoolSchema
from dr_llm.pool.runtime import DbConfig, DbRuntime
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

    _wait_for_terminal_queue(fill_store)
    controller.stop()
    snapshot = controller.join()

    assert snapshot.claimed == 8
    assert snapshot.promoted == 8
    assert snapshot.failed == 0
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

    _wait_for_terminal_queue(fill_store)
    controller.stop()
    snapshot = controller.join()

    assert snapshot.claimed == 2
    assert snapshot.retried == 1
    assert snapshot.failed == 1
    assert snapshot.process_errors == 2
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

    _wait_for_terminal_queue(fill_store)
    controller.stop()
    snapshot = controller.join()

    assert snapshot.claimed == 2
    assert snapshot.retried == 1
    assert snapshot.failed == 0
    assert snapshot.promoted == 1
    assert snapshot.process_errors == 1
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

    _wait_for_terminal_queue(fill_store, key_filter={"model": "m1"})
    controller.stop()
    snapshot = controller.join()
    all_counts = fill_store.pending.status_counts()

    assert snapshot.promoted == 2
    assert snapshot.status_counts.pending == 0
    assert snapshot.status_counts.promoted == 2
    assert all_counts.pending == 2
    assert all_counts.promoted == 2
    assert fill_store.cell_depth(key_values={"model": "m1", "prompt": "shared"}) == 2
    assert fill_store.cell_depth(key_values={"model": "m2", "prompt": "shared"}) == 0


@pytest.mark.integration
def test_demo_pool_fill_script_runs() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    try:
        with psycopg.connect(dsn):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unavailable for pool fill integration tests: {exc}")

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "demo-pool-fill.py"
    pool_name = f"itest_demo_fill_{uuid4().hex[:8]}"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(script_path),
            "--dsn",
            dsn,
            "--pool-name",
            pool_name,
            "--num-workers",
            "2",
            "--samples-per-cell",
            "2",
            "--models",
            "demo-a",
            "--models",
            "demo-b",
            "--prompts",
            "alpha",
            "--prompts",
            "beta",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Seeded 8 pending rows" in result.stdout
    assert "Final queue counts: pending=0 leased=0 promoted=8 failed=0" in result.stdout
    assert "Stored 8 samples across 4 cells" in result.stdout
