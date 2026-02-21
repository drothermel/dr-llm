from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from llm_pool.errors import SessionConflictError
from llm_pool.providers.base import ProviderCapabilities
from llm_pool.session.client import SessionClient
from llm_pool.tools.registry import ToolRegistry
from llm_pool.types import (
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


@dataclass
class FakeAdapter:
    capabilities: ProviderCapabilities


@dataclass
class FakeLlmClient:
    adapter: FakeAdapter
    response: LlmResponse

    def get_adapter(self, provider_name: str):  # noqa: ARG002
        return self.adapter

    def query(self, request, **kwargs):  # noqa: ANN001, ARG002
        return self.response


@dataclass
class FakeRepository:
    session_state: SessionState
    events: list[tuple[str, dict]] = field(default_factory=list)
    enqueued: list[str] = field(default_factory=list)
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
    ):  # noqa: ARG002
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
    ):  # noqa: ARG002
        persisted_id = tool_call_id or "queued_call"
        self.enqueued.append(persisted_id)
        return persisted_id

    def complete_tool_call(self, *, result):  # noqa: ANN001, ARG002
        raise AssertionError("Should not complete tools in queued brokered mode")

    def complete_session_turn(self, *, turn_id: str, status):  # noqa: ANN001, ARG002
        pass

    def update_session_status(self, *, session_id: str, status, last_error_text=None):  # noqa: ANN001, ARG002
        pass


class RaisingExecutor:
    def invoke(self, call):  # noqa: ANN001
        raise AssertionError("Tool executor should not run for queued brokered mode")


def _session_state(status: SessionStatus = SessionStatus.active) -> SessionState:
    now = utcnow()
    return SessionState(
        session_id="s1",
        status=status,
        version=1,
        strategy_mode=ToolPolicy.brokered_only,
        metadata={"provider": "openai", "model": "gpt-test"},
        created_at=now,
        updated_at=now,
    )


def test_step_rejects_non_active_session() -> None:
    repo = FakeRepository(session_state=_session_state(SessionStatus.canceled))
    client = SessionClient(
        llm_client=FakeLlmClient(
            adapter=FakeAdapter(ProviderCapabilities(supports_native_tools=False)),
            response=LlmResponse(
                text="",
                usage=TokenUsage(),
                provider="openai",
                model="gpt-test",
                mode=CallMode.api,
            ),
        ),
        repository=repo,
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
        adapter=FakeAdapter(ProviderCapabilities(supports_native_tools=False)),
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
        llm_client=llm_client,
        repository=repo,
        tool_registry=ToolRegistry(),
        tool_executor=RaisingExecutor(),
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
