from __future__ import annotations

from datetime import datetime, timezone

from dr_llm.catalog.models import ModelCatalogQuery
from dr_llm.pool.recorded_call import RecordedCall, RunStatus
from dr_llm.providers.models import CallMode


def test_model_catalog_query_default_limit() -> None:
    assert ModelCatalogQuery().limit == 200


def test_recorded_call_coerces_status_to_enum() -> None:
    call = RecordedCall(
        call_id="call_123",
        run_id="run_123",
        provider="openai",
        model="gpt-4.1",
        mode=CallMode.api,
        status=RunStatus.success,
        created_at=datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc),
        latency_ms=12,
        error_text=None,
        request={},
        response=None,
    )

    assert call.status is RunStatus.success
