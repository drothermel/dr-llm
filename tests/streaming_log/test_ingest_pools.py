from __future__ import annotations

import asyncio
from typing import cast

import pytest

from dr_llm.pool.pool_sample import PoolSample
from dr_llm.streaming_log import StreamingLogClient
from dr_llm.streaming_log.ingest_pools import (
    _sample_snapshot_payload,
    ingest_pool,
    ingest_pools,
)


def test_sample_snapshot_payload_preserves_pool_row_state() -> None:
    sample = PoolSample(
        sample_id="sample-1",
        key_values={"dim": "a"},
        sample_idx=3,
        run_id="run-1",
        request={"prompt": "hello"},
        response={"text": "world"},
        finish_reason="stop",
        attempt_count=2,
        metadata={"m": 1},
    )

    payload = _sample_snapshot_payload(sample)

    assert payload["sample_id"] == "sample-1"
    assert payload["key_values"] == {"dim": "a"}
    assert payload["sample_idx"] == 3
    assert payload["run_id"] == "run-1"
    assert payload["request"] == {"prompt": "hello"}
    assert payload["response"] == {"text": "world"}
    assert payload["finish_reason"] == "stop"
    assert payload["attempt_count"] == 2
    assert payload["metadata"] == {"m": 1}


def test_ingest_pool_rejects_non_positive_sample_limit() -> None:
    with pytest.raises(ValueError, match="sample_limit must be at least 1"):
        asyncio.run(
            ingest_pool(
                client=cast(StreamingLogClient, None),
                dsn="postgresql://unused",
                pool_name="unused",
                sample_limit=0,
            )
        )


def test_ingest_pools_rejects_non_positive_sample_limit() -> None:
    with pytest.raises(ValueError, match="sample_limit must be at least 1"):
        asyncio.run(
            ingest_pools(
                client=cast(StreamingLogClient, None),
                dsn="postgresql://unused",
                sample_limit=0,
            )
        )
