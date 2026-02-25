from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import SessionConflictError
from dr_llm.providers.base import ProviderCapabilities
from dr_llm.session.client import SessionClient
from dr_llm.tools.registry import ToolRegistry
from dr_llm.types import (
    CallMode,
    LlmResponse,
    Message,
    ModelToolCall,
    SessionState,
    SessionStatus,
    SessionStepInput,
    ToolPolicy,
    TokenUsage,
    utcnow,
)


class FakeAdapter(BaseModel):
    capabilities: ProviderCapabilities


class FakeLlmClient(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: FakeAdapter
    response: LlmResponse

    def get_adapter(self, provider_name: str):  # noqa: ARG002
        return self.adapter

    def query(self, request, **kwargs):  # noqa: ANN001, ARG002
        return self.response


class FakeRepository(BaseModel):
    session_state: SessionState
    events: list[tuple[str, dict[str, Any]]] = Field(default_factory=list)
    enqueued: list[str] = Field(default_factory=list)
    advanced: bool = False

    def get_session(self, *, session_id: str):  # noqa: ARG002
        return self.session_state

    def advance_session_version(self, *, session_id: str, expected_version: int):  # noqa: ARG002
        self.advanced = True
        return expected_version + 1

    def create_session_turn(self, *, session_id: str, status, metadata):  # noqa: ANN001, ARG002
        return "turn_1", 0

    def replay_session_messages(self, *, session_id: str):  # noqa: ARG002
        return []

    def append_session_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict,
        turn_id: str | None = None,
    ):
        self.events.append((event_type, payload))
        return "event_1"

    def enqueue_tool_call(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict,
        idempotency_key: str,
        turn_id: str | None = None,
        tool_call_id: str | None = None,
    ):
        persisted_id = tool_call_id or "queued_call"
        self.enqueued.append(persisted_id)
        return persisted_id

    def complete_tool_call(self, *, result):
        _ = result
        raise AssertionError("Should not complete tools in queued brokered mode")

    def complete_session_turn(self, *, turn_id: str, status):
        _ = (turn_id, status)

    def update_session_status(self, *, session_id: str, status, last_error_text=None):
        _ = (session_id, status, last_error_text)


class RaisingExecutor:
    def invoke(self, call):  # noqa: ANN001
        raise AssertionError("Tool executor should not run for queued brokered mode")


def _session_state(
    status: SessionStatus = SessionStatus.active,
    strategy_mode: ToolPolicy = ToolPolicy.brokered_only,
) -> SessionState:
    now = utcnow()
    return SessionState(
        session_id="s1",
        status=status,
        version=1,
        strategy_mode=strategy_mode,
        metadata={"provider": "openai", "model": "gpt-test"},
        created_at=now,
        updated_at=now,
    )


def test_step_rejects_non_active_session() -> None:
    repo = FakeRepository(session_state=_session_state(SessionStatus.canceled))
    client = SessionClient(
        llm_client=cast(
            Any,
            FakeLlmClient(
                adapter=FakeAdapter(
                    capabilities=ProviderCapabilities(supports_native_tools=False)
                ),
                response=LlmResponse(
                    text="",
                    usage=TokenUsage(),
                    provider="openai",
                    model="gpt-test",
                    mode=CallMode.api,
                ),
            ),
        ),
        repository=cast(Any, repo),
        tool_registry=ToolRegistry(),
    )

    with pytest.raises(SessionConflictError):
        client.step_session(
            SessionStepInput(
                session_id="s1", messages=[Message(role="user", content="next")]
            )
        )

    assert repo.advanced is False


def test_brokered_step_queues_tools_when_inline_disabled() -> None:
    repo = FakeRepository(session_state=_session_state(SessionStatus.active))
    llm_client = FakeLlmClient(
        adapter=FakeAdapter(
            capabilities=ProviderCapabilities(supports_native_tools=False)
        ),
        response=LlmResponse(
            text='{"tool_calls":[{"name":"lookup","arguments":{"q":"x"}}]}',
            usage=TokenUsage(),
            provider="openai",
            model="gpt-test",
            mode=CallMode.api,
            tool_calls=[
                ModelToolCall(tool_call_id="tc_1", name="lookup", arguments={"q": "x"})
            ],
        ),
    )
    client = SessionClient(
        llm_client=cast(Any, llm_client),
        repository=cast(Any, repo),
        tool_registry=ToolRegistry(),
        tool_executor=cast(Any, RaisingExecutor()),
    )

    result = client.step_session(
        SessionStepInput(
            session_id="s1",
            messages=[Message(role="user", content="go")],
            inline_tool_execution=False,
        )
    )

    assert result.status.value == "active"
    assert repo.enqueued == ["tc_1"]
    assert any(event == "session_waiting_for_tools" for event, _ in repo.events)


def test_native_only_with_unsupported_provider_raises_value_error() -> None:
    repo = FakeRepository(
        session_state=_session_state(
            SessionStatus.active,
            strategy_mode=ToolPolicy.native_only,
        )
    )
    llm_client = FakeLlmClient(
        adapter=FakeAdapter(
            capabilities=ProviderCapabilities(supports_native_tools=False)
        ),
        response=LlmResponse(
            text="",
            usage=TokenUsage(),
            provider="openai",
            model="gpt-test",
            mode=CallMode.api,
            tool_calls=[
                ModelToolCall(tool_call_id="tc_1", name="lookup", arguments={"q": "x"})
            ],
        ),
    )
    client = SessionClient(
        llm_client=cast(Any, llm_client),
        repository=cast(Any, repo),
        tool_registry=ToolRegistry(),
        tool_executor=cast(Any, RaisingExecutor()),
    )

    with pytest.raises(
        ValueError,
        match="Session configured native_only, but provider does not support native tools",
    ):
        client.step_session(
            SessionStepInput(
                session_id="s1",
                messages=[Message(role="user", content="go")],
            )
        )
