from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dr_llm.catalog.models import ModelCatalogQuery
from dr_llm.providers.models import CallMode, Message
from dr_llm.providers.reasoning import ReasoningConfig
from dr_llm.providers.usage import TokenUsage
from dr_llm.storage.models import RecordedCall, RunStatus


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="tool", content="nope")  # type: ignore[arg-type]


def test_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Message(role="assistant", content="hi", tool_calls=[])  # type: ignore[call-arg]


def test_reasoning_config_rejects_effort_with_max_tokens() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(effort="high", max_tokens=100)


def test_reasoning_config_effective_enabled() -> None:
    assert ReasoningConfig(effort="low").effective_enabled
    assert not ReasoningConfig(enabled=False).effective_enabled


def test_token_usage_computed_total_and_non_negative_validation() -> None:
    usage = TokenUsage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=10,
        reasoning_tokens=1,
    )
    assert usage.computed_total_tokens == 5

    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1)


def test_token_usage_coerces_and_derives_total() -> None:
    usage = TokenUsage.from_raw(
        prompt_tokens="4", completion_tokens="6", total_tokens=None
    )
    assert usage.prompt_tokens == 4
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 10


def test_token_usage_rejects_invalid_integer_like_values() -> None:
    with pytest.raises(ValidationError):
        TokenUsage.from_raw(prompt_tokens="abc")


def test_model_catalog_query_default_limit_remains_non_cli() -> None:
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
