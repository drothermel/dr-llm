"""Integration tests for dr_llm.backends PoolBackend (requires PostgreSQL)."""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg import sql

from dr_llm.backends.converters import backend_request_payload
from dr_llm.backends.fingerprint import fingerprint_request
from dr_llm.backends.models import BackendRequest, PoolBackendConfig
from dr_llm.backends.pool import PoolBackend
from dr_llm.backends.schema import BACKENDS_KEY_COLUMN, backends_pool_schema
from dr_llm.errors import TransientPersistenceError
from dr_llm.llm import CallMode, LlmResponse, Message, ProviderName, TokenUsage
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.sampling.db.names import claims_table_name

_POOL_NAME = "itest_backends"
_SCHEMA = backends_pool_schema(_POOL_NAME)
_CONSUMER_ID = _POOL_NAME


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv(
        "DR_LLM_DATABASE_URL"
    )


def _drop_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for tbl in reversed(_SCHEMA.table_names()):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", tbl)
                )
            )
        claims_tbl = claims_table_name(_SCHEMA.name, _CONSUMER_ID)
        conn.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier("public", claims_tbl)
            )
        )
        catalog_exists = conn.execute(
            "SELECT to_regclass('public.pool_catalog') IS NOT NULL"
        ).fetchone()[0]
        if catalog_exists:
            conn.execute(
                "DELETE FROM pool_catalog WHERE pool_name = %s",
                [_POOL_NAME],
            )
        conn.commit()


def _request(
    model: str = "gpt-4.1-mini",
    *,
    content: str = "integration test prompt",
) -> BackendRequest:
    return BackendRequest(
        provider=ProviderName.OPENAI,
        model=model,
        mode=CallMode.api,
        messages=[Message(role="user", content=content)],
    )


def _llm_response(*, text: str, model: str = "gpt-4.1-mini") -> LlmResponse:
    return LlmResponse(
        text=text,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        provider=ProviderName.OPENAI,
        model=model,
        mode=CallMode.api,
    )


def _mock_registry() -> MagicMock:
    counter = {"n": 0}

    def _generate(_request: object) -> LlmResponse:
        counter["n"] += 1
        return _llm_response(text=f"generated-{counter['n']}")

    orchestrator = MagicMock()
    orchestrator.generate.side_effect = _generate
    registry = MagicMock()
    registry.get.return_value = orchestrator
    return registry


@pytest.fixture(scope="module")
def pool_backend() -> Generator[PoolBackend, None, None]:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip(
            "Set DR_LLM_TEST_DATABASE_URL to run backends integration tests"
        )
    try:
        _drop_tables(dsn)
        backend = PoolBackend(
            PoolBackendConfig(
                pool_name=_POOL_NAME,
                database_url=dsn,
                consumer_id=_CONSUMER_ID,
                num_workers=2,
                lease_seconds=30,
                acquire_timeout_seconds=30,
            ),
            registry=_mock_registry(),
        )
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        pytest.skip(f"Postgres unavailable for backends integration tests: {exc}")
    yield backend
    backend.close()
    dsn_val = _get_dsn()
    if dsn_val:
        _drop_tables(dsn_val)


def _seed_complete_samples(
    store: PoolStore,
    request: BackendRequest,
    *,
    count: int,
) -> str:
    fingerprint = fingerprint_request(request)
    for idx in range(count):
        store.insert_sample(
            PoolSample(
                key_values={BACKENDS_KEY_COLUMN: fingerprint},
                request=backend_request_payload(request),
                response=_llm_response(text=f"seed-{idx}").model_dump(
                    mode="json"
                ),
                finish_reason="stop",
                sample_idx=idx,
            )
        )
    return fingerprint


@pytest.mark.integration
def test_acquire_session_semantics_12_10_3(pool_backend: PoolBackend) -> None:
    request = _request(content="acquire session semantics prompt")
    _seed_complete_samples(pool_backend.store, request, count=12)

    first = pool_backend.acquire(request, session_id="s1", n=10)
    assert len(first.responses) == 10
    assert first.claimed_from_cache == 10
    assert first.generated == 0

    second = pool_backend.acquire(request, session_id="s1", n=3)
    assert len(second.responses) == 3
    assert second.claimed_from_cache == 2
    assert second.generated == 1

    third = pool_backend.acquire(request, session_id="s2", n=3)
    assert len(third.responses) == 3
    assert third.claimed_from_cache == 3
    assert third.generated == 0


@pytest.mark.integration
def test_submit_batch_and_drain(pool_backend: PoolBackend) -> None:
    request_a = _request(
        model="gpt-4.1-mini",
        content="submit batch prompt a",
    )
    request_b = _request(
        model="gpt-4.1",
        content="submit batch prompt b",
    )

    complete_before = pool_backend.store.complete_count()

    submit = pool_backend.submit_batch([request_a, request_b])
    assert submit.seeded == 2
    assert submit.skipped == 0
    assert pool_backend.store.incomplete_count() == 2

    drain = pool_backend.await_drain(timeout=30)
    assert drain.incomplete == 0
    assert pool_backend.store.complete_count() == complete_before + 2
    assert drain.worker_counts.completed == 2

    hit_a = pool_backend.complete(request_a)
    hit_b = pool_backend.complete(request_b)
    assert hit_a.source == "pool_cache"
    assert hit_b.source == "pool_cache"
    assert hit_a.text.startswith("generated-")
    assert hit_b.text.startswith("generated-")
