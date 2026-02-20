from __future__ import annotations

import json

from llm_pool.client import LlmClient
from llm_pool.session.strategy import resolve_tool_strategy
from llm_pool.storage.repository import PostgresRepository
from llm_pool.tools.executor import ToolExecutor
from llm_pool.tools.registry import ToolRegistry
from llm_pool.types import (
    LlmRequest,
    Message,
    ModelToolCall,
    SessionHandle,
    SessionStartInput,
    SessionState,
    SessionStepInput,
    SessionStepResult,
    SessionStatus,
    SessionTurnStatus,
    ToolInvocation,
)


def _parse_brokered_tool_calls(text: str) -> list[ModelToolCall]:
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    parsed: list[ModelToolCall] = []
    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        args = item.get("arguments")
        if not isinstance(args, dict):
            args = {}
        parsed.append(
            ModelToolCall(
                tool_call_id=str(item.get("tool_call_id") or f"brokered_call_{idx+1}"),
                name=name,
                arguments=args,
            )
        )
    return parsed


class SessionClient:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        repository: PostgresRepository,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._repository = repository
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(tool_registry)

    def start_session(self, input: SessionStartInput) -> SessionHandle:
        metadata = {
            **input.metadata,
            "provider": input.provider,
            "model": input.model,
            "run_id": input.run_id,
        }
        handle = self._repository.start_session(
            strategy_mode=input.strategy_mode,
            metadata=metadata,
        )
        turn_id, _turn_index = self._repository.create_session_turn(
            session_id=handle.session_id,
            status=SessionTurnStatus.active,
            metadata={"kind": "session_start"},
        )
        self._repository.append_session_event(
            session_id=handle.session_id,
            turn_id=turn_id,
            event_type="session_started",
            payload={"messages": [message.model_dump(mode="json") for message in input.messages]},
        )
        for message in input.messages:
            self._repository.append_session_event(
                session_id=handle.session_id,
                turn_id=turn_id,
                event_type="message",
                payload={"message": message.model_dump(mode="json")},
            )
        self._repository.complete_session_turn(turn_id=turn_id, status=SessionTurnStatus.completed)
        return handle

    def resume_session(self, session_id: str) -> SessionState:
        return self._repository.get_session(session_id=session_id)

    def cancel_session(self, session_id: str, reason: str) -> None:
        self._repository.append_session_event(
            session_id=session_id,
            event_type="session_canceled",
            payload={"reason": reason},
        )
        self._repository.update_session_status(
            session_id=session_id,
            status=SessionStatus.canceled,
            last_error_text=reason,
        )

    def step_session(self, input: SessionStepInput) -> SessionStepResult:
        state = self._repository.get_session(session_id=input.session_id)
        expected_version = input.expected_version if input.expected_version is not None else state.version
        new_version = self._repository.advance_session_version(
            session_id=input.session_id,
            expected_version=expected_version,
        )
        turn_id, turn_index = self._repository.create_session_turn(
            session_id=input.session_id,
            status=SessionTurnStatus.active,
            metadata={"kind": "session_step"},
        )

        try:
            history = self._repository.replay_session_messages(session_id=input.session_id)
            conversation = [Message.model_validate(m) for m in history]
            conversation.extend(input.messages)

            for message in input.messages:
                self._repository.append_session_event(
                    session_id=input.session_id,
                    turn_id=turn_id,
                    event_type="message",
                    payload={"message": message.model_dump(mode="json")},
                )

            adapter = self._llm_client.get_adapter(state.metadata.get("provider", ""))
            strategy = resolve_tool_strategy(
                policy=state.strategy_mode,
                capabilities=adapter.capabilities,
            )

            provider = str(state.metadata.get("provider"))
            model = str(state.metadata.get("model"))
            run_id = (
                str(state.metadata.get("run_id"))
                if state.metadata.get("run_id") is not None
                else None
            )
            if not provider or not model:
                raise ValueError(
                    f"Session metadata missing provider/model for session={input.session_id}"
                )

            request = LlmRequest(
                provider=provider,
                model=model,
                messages=conversation,
                metadata=input.metadata,
                tools=self._tool_registry.to_provider_tools() or None,
                tool_policy=state.strategy_mode,
            )

            initial_response = self._llm_client.query(
                request,
                run_id=run_id,
                external_call_id=f"session:{input.session_id}:turn:{turn_id}:initial",
                metadata={"session_id": input.session_id, "turn_id": turn_id},
            )
            self._repository.append_session_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type="model_response",
                payload={
                    "response": {
                        "text": initial_response.text,
                        "finish_reason": initial_response.finish_reason,
                        "tool_calls": [tc.model_dump(mode="json") for tc in initial_response.tool_calls],
                    }
                },
            )

            tool_calls = list(initial_response.tool_calls)
            if not tool_calls and strategy == "brokered":
                tool_calls = _parse_brokered_tool_calls(initial_response.text)

            final_output = Message(role="assistant", content=initial_response.text)
            if tool_calls:
                if strategy == "native" and state.strategy_mode.value == "native_only" and not adapter.capabilities.supports_native_tools:
                    raise ValueError("Session configured native_only, but provider does not support native tools")

                assistant_tool_request_text = initial_response.text.strip()
                if assistant_tool_request_text:
                    conversation.append(Message(role="assistant", content=assistant_tool_request_text))
                    self._repository.append_session_event(
                        session_id=input.session_id,
                        turn_id=turn_id,
                        event_type="message",
                        payload={"message": Message(role="assistant", content=assistant_tool_request_text).model_dump(mode="json")},
                    )

                for model_tool_call in tool_calls:
                    self._repository.append_session_event(
                        session_id=input.session_id,
                        turn_id=turn_id,
                        event_type="model_requested_tool",
                        payload={"tool_call": model_tool_call.model_dump(mode="json")},
                    )
                    idempotency_key = (
                        f"{input.session_id}:{turn_id}:{model_tool_call.tool_call_id}:{model_tool_call.name}"
                    )
                    persisted_tool_call_id = self._repository.enqueue_tool_call(
                        session_id=input.session_id,
                        turn_id=turn_id,
                        tool_name=model_tool_call.name,
                        args=model_tool_call.arguments,
                        idempotency_key=idempotency_key,
                        tool_call_id=model_tool_call.tool_call_id,
                    )
                    self._repository.append_session_event(
                        session_id=input.session_id,
                        turn_id=turn_id,
                        event_type="tool_started",
                        payload={
                            "tool_call_id": persisted_tool_call_id,
                            "tool_name": model_tool_call.name,
                        },
                    )
                    tool_result = self._tool_executor.invoke(
                        ToolInvocation(
                            tool_call_id=persisted_tool_call_id,
                            name=model_tool_call.name,
                            arguments=model_tool_call.arguments,
                            session_id=input.session_id,
                            turn_id=turn_id,
                        )
                    )
                    self._repository.complete_tool_call(result=tool_result)
                    event_type = "tool_succeeded" if tool_result.ok else "tool_failed"
                    self._repository.append_session_event(
                        session_id=input.session_id,
                        turn_id=turn_id,
                        event_type=event_type,
                        payload={
                            "tool_call_id": persisted_tool_call_id,
                            "tool_name": model_tool_call.name,
                            "result": tool_result.result,
                            "error": tool_result.error,
                        },
                    )
                    tool_message_content = json.dumps(
                        tool_result.result if tool_result.ok else {"error": tool_result.error},
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    tool_message = Message(
                        role="tool",
                        name=model_tool_call.name,
                        content=tool_message_content,
                    )
                    conversation.append(tool_message)
                    self._repository.append_session_event(
                        session_id=input.session_id,
                        turn_id=turn_id,
                        event_type="tool_result_message",
                        payload={"message": tool_message.model_dump(mode="json")},
                    )

                followup_request = LlmRequest(
                    provider=provider,
                    model=model,
                    messages=conversation,
                    metadata={
                        **input.metadata,
                        "followup_after_tools": True,
                    },
                    tools=self._tool_registry.to_provider_tools() or None,
                    tool_policy=state.strategy_mode,
                )
                followup_response = self._llm_client.query(
                    followup_request,
                    run_id=run_id,
                    external_call_id=f"session:{input.session_id}:turn:{turn_id}:followup",
                    metadata={"session_id": input.session_id, "turn_id": turn_id, "phase": "followup"},
                )
                final_output = Message(role="assistant", content=followup_response.text)

            self._repository.append_session_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type="model_completed",
                payload={"message": final_output.model_dump(mode="json")},
            )
            self._repository.append_session_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type="message",
                payload={"message": final_output.model_dump(mode="json")},
            )
            self._repository.complete_session_turn(
                turn_id=turn_id,
                status=SessionTurnStatus.completed,
            )
            return SessionStepResult(
                session_id=input.session_id,
                turn_id=turn_id,
                status=SessionTurnStatus.completed,
                version=new_version,
                output=final_output,
                tool_calls=tool_calls,
            )
        except Exception as exc:  # noqa: BLE001
            self._repository.append_session_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type="session_step_failed",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
            )
            self._repository.complete_session_turn(
                turn_id=turn_id,
                status=SessionTurnStatus.failed,
            )
            self._repository.update_session_status(
                session_id=input.session_id,
                status=SessionStatus.failed,
                last_error_text=str(exc),
            )
            raise
