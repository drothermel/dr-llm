from __future__ import annotations

import json
from typing import Any
from typing import Literal

from pydantic import BaseModel, ValidationError

from llm_pool.client import LlmClient
from llm_pool.errors import SessionConflictError
from llm_pool.session.models import (
    ModelRequestedToolPayload,
    ModelResponseData,
    ModelResponsePayload,
    SessionCanceledPayload,
    SessionEventType,
    SessionMessagePayload,
    SessionMetadata,
    SessionStartedPayload,
    SessionStepFailedPayload,
    SessionWaitingForToolsPayload,
    ToolExecutionPayload,
    ToolLifecyclePayload,
    ToolProcessingResult,
    error_payload,
    parse_brokered_tool_calls,
    payload_dict,
)
from llm_pool.session.strategy import resolve_tool_strategy
from llm_pool.storage.repository import PostgresRepository
from llm_pool.tools.executor import ToolExecutor
from llm_pool.tools.registry import ToolRegistry
from llm_pool.types import (
    LlmRequest,
    LlmResponse,
    Message,
    ModelToolCall,
    ReasoningConfig,
    SessionHandle,
    SessionStartInput,
    SessionState,
    SessionStatus,
    SessionStepInput,
    SessionStepResult,
    SessionTurnStatus,
    ToolInvocation,
    ToolPolicy,
)


def _parse_brokered_tool_calls(text: str) -> list[ModelToolCall]:
    return parse_brokered_tool_calls(text)


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
        self._tool_executor = tool_executor or ToolExecutor(registry=tool_registry)

    def _append_event(
        self,
        *,
        session_id: str,
        event_type: SessionEventType,
        payload: BaseModel,
        turn_id: str | None = None,
    ) -> None:
        self._repository.append_session_event(
            session_id=session_id,
            turn_id=turn_id,
            event_type=event_type.value,
            payload=payload_dict(payload),
        )

    def _load_session_metadata(self, *, state: SessionState) -> SessionMetadata:
        try:
            return SessionMetadata(**state.metadata)
        except ValidationError as exc:
            raise ValueError(
                f"Session metadata missing provider/model for session={state.session_id}"
            ) from exc

    def _build_start_metadata(self, *, input: SessionStartInput) -> dict[str, Any]:
        extra_metadata = {
            key: value
            for key, value in input.metadata.items()
            if key not in {"provider", "model", "run_id", "reasoning"}
        }
        metadata = SessionMetadata(
            provider=input.provider,
            model=input.model,
            run_id=input.run_id,
            reasoning=input.reasoning,
            **extra_metadata,
        )
        return metadata.model_dump(mode="json", exclude_computed_fields=True)

    def _record_messages(
        self,
        *,
        session_id: str,
        turn_id: str,
        messages: list[Message],
    ) -> None:
        for message in messages:
            self._append_event(
                session_id=session_id,
                turn_id=turn_id,
                event_type=SessionEventType.message,
                payload=SessionMessagePayload(message=message),
            )

    def _record_model_response(
        self,
        *,
        session_id: str,
        turn_id: str,
        response: LlmResponse,
    ) -> None:
        self._append_event(
            session_id=session_id,
            turn_id=turn_id,
            event_type=SessionEventType.model_response,
            payload=ModelResponsePayload(
                response=ModelResponseData(
                    text=response.text,
                    finish_reason=response.finish_reason,
                    tool_calls=response.tool_calls,
                )
            ),
        )

    def _record_step_failure(
        self,
        *,
        session_id: str,
        turn_id: str,
        exc: Exception,
    ) -> None:
        self._append_event(
            session_id=session_id,
            turn_id=turn_id,
            event_type=SessionEventType.session_step_failed,
            payload=SessionStepFailedPayload(
                error_type=type(exc).__name__,
                message=str(exc),
            ),
        )
        self._repository.complete_session_turn(
            turn_id=turn_id,
            status=SessionTurnStatus.failed,
        )
        self._repository.update_session_status(
            session_id=session_id,
            status=SessionStatus.failed,
            last_error_text=str(exc),
        )

    def _resolve_tool_calls(
        self,
        *,
        response: LlmResponse,
        strategy: Literal["brokered", "native"],
    ) -> list[ModelToolCall]:
        tool_calls = list(response.tool_calls)
        if not tool_calls and strategy == "brokered":
            return parse_brokered_tool_calls(response.text)
        return tool_calls

    def _handle_tool_calls(
        self,
        *,
        state: SessionState,
        input: SessionStepInput,
        session_metadata: SessionMetadata,
        turn_id: str,
        strategy: Literal["brokered", "native"],
        supports_native_tools: bool,
        resolved_reasoning: ReasoningConfig | None,
        conversation: list[Message],
        tool_calls: list[ModelToolCall],
        initial_response: LlmResponse,
    ) -> ToolProcessingResult:
        assistant_tool_request = Message(
            role="assistant",
            content=initial_response.text,
            tool_calls=tool_calls if strategy == "native" else None,
        )
        conversation.append(assistant_tool_request)
        self._append_event(
            session_id=input.session_id,
            turn_id=turn_id,
            event_type=SessionEventType.message,
            payload=SessionMessagePayload(message=assistant_tool_request),
        )

        if (
            strategy == "native"
            and state.strategy_mode == ToolPolicy.native_only
            and not supports_native_tools
        ):
            raise ValueError(
                "Session configured native_only, but provider does not support native tools"
            )

        execute_inline = strategy == "native" or input.inline_tool_execution
        for model_tool_call in tool_calls:
            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=SessionEventType.model_requested_tool,
                payload=ModelRequestedToolPayload(tool_call=model_tool_call),
            )
            idempotency_key = f"{input.session_id}:{turn_id}:{model_tool_call.tool_call_id}:{model_tool_call.name}"
            persisted_tool_call_id = self._repository.enqueue_tool_call(
                session_id=input.session_id,
                turn_id=turn_id,
                tool_name=model_tool_call.name,
                args=model_tool_call.arguments,
                idempotency_key=idempotency_key,
                tool_call_id=model_tool_call.tool_call_id,
            )
            tool_lifecycle_payload = ToolLifecyclePayload(
                tool_call_id=persisted_tool_call_id,
                tool_name=model_tool_call.name,
            )
            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=SessionEventType.tool_started,
                payload=tool_lifecycle_payload,
            )
            if not execute_inline:
                self._append_event(
                    session_id=input.session_id,
                    turn_id=turn_id,
                    event_type=SessionEventType.tool_queued,
                    payload=tool_lifecycle_payload,
                )
                continue

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

            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=(
                    SessionEventType.tool_succeeded
                    if tool_result.ok
                    else SessionEventType.tool_failed
                ),
                payload=ToolExecutionPayload(
                    tool_call_id=persisted_tool_call_id,
                    tool_name=model_tool_call.name,
                    result=tool_result.result,
                    error=tool_result.error,
                ),
            )

            tool_message_content = json.dumps(
                tool_result.result
                if tool_result.ok
                else {"error": error_payload(tool_result.error)},
                ensure_ascii=True,
                sort_keys=True,
            )
            tool_message = Message(
                role="tool",
                name=model_tool_call.name,
                content=tool_message_content,
                tool_call_id=model_tool_call.tool_call_id,
            )
            conversation.append(tool_message)
            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=SessionEventType.tool_result_message,
                payload=SessionMessagePayload(message=tool_message),
            )

        if not execute_inline:
            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=SessionEventType.session_waiting_for_tools,
                payload=SessionWaitingForToolsPayload(
                    tool_call_ids=[call.tool_call_id for call in tool_calls],
                    message="Tool calls enqueued; run worker and step session again to continue.",
                ),
            )
            return ToolProcessingResult(
                final_output=assistant_tool_request,
                tool_calls=tool_calls,
                waiting_for_tools=True,
            )

        followup_request = LlmRequest(
            provider=session_metadata.provider,
            model=session_metadata.model,
            messages=conversation,
            reasoning=resolved_reasoning,
            metadata={
                **input.metadata,
                "followup_after_tools": True,
            },
            tools=self._tool_registry.to_provider_tools() or None,
            tool_policy=state.strategy_mode,
        )
        followup_response = self._llm_client.query(
            followup_request,
            run_id=session_metadata.run_id,
            external_call_id=f"session:{input.session_id}:turn:{turn_id}:followup",
            metadata={
                "session_id": input.session_id,
                "turn_id": turn_id,
                "phase": "followup",
            },
        )
        return ToolProcessingResult(
            final_output=Message(role="assistant", content=followup_response.text),
            tool_calls=tool_calls,
        )

    def start_session(self, input: SessionStartInput) -> SessionHandle:
        metadata = self._build_start_metadata(input=input)
        handle = self._repository.start_session(
            strategy_mode=input.strategy_mode,
            metadata=metadata,
        )
        turn_id, _turn_index = self._repository.create_session_turn(
            session_id=handle.session_id,
            status=SessionTurnStatus.active,
            metadata={"kind": "session_start"},
        )
        self._append_event(
            session_id=handle.session_id,
            turn_id=turn_id,
            event_type=SessionEventType.session_started,
            payload=SessionStartedPayload(messages=input.messages),
        )
        self._record_messages(
            session_id=handle.session_id,
            turn_id=turn_id,
            messages=input.messages,
        )
        self._repository.complete_session_turn(
            turn_id=turn_id, status=SessionTurnStatus.completed
        )
        return handle

    def resume_session(self, session_id: str) -> SessionState:
        return self._repository.get_session(session_id=session_id)

    def cancel_session(self, session_id: str, reason: str) -> None:
        self._append_event(
            session_id=session_id,
            event_type=SessionEventType.session_canceled,
            payload=SessionCanceledPayload(reason=reason),
        )
        self._repository.update_session_status(
            session_id=session_id,
            status=SessionStatus.canceled,
            last_error_text=reason,
        )

    def step_session(self, input: SessionStepInput) -> SessionStepResult:
        state = self._repository.get_session(session_id=input.session_id)
        if state.status != SessionStatus.active:
            raise SessionConflictError(
                f"Cannot step session {input.session_id}: status={state.status.value}"
            )
        session_metadata = self._load_session_metadata(state=state)

        expected_version = (
            input.expected_version
            if input.expected_version is not None
            else state.version
        )
        new_version = self._repository.advance_session_version(
            session_id=input.session_id,
            expected_version=expected_version,
        )
        turn_id, _turn_index = self._repository.create_session_turn(
            session_id=input.session_id,
            status=SessionTurnStatus.active,
            metadata={"kind": "session_step"},
        )

        try:
            history = self._repository.replay_session_messages(
                session_id=input.session_id
            )
            conversation = [Message(**message) for message in history]
            conversation.extend(input.messages)
            self._record_messages(
                session_id=input.session_id,
                turn_id=turn_id,
                messages=input.messages,
            )

            adapter = self._llm_client.get_adapter(session_metadata.provider)
            strategy = resolve_tool_strategy(
                policy=state.strategy_mode,
                capabilities=adapter.capabilities,
            )
            resolved_reasoning = input.reasoning or session_metadata.reasoning

            request = LlmRequest(
                provider=session_metadata.provider,
                model=session_metadata.model,
                messages=conversation,
                metadata=input.metadata,
                reasoning=resolved_reasoning,
                tools=self._tool_registry.to_provider_tools() or None,
                tool_policy=state.strategy_mode,
            )
            initial_response = self._llm_client.query(
                request,
                run_id=session_metadata.run_id,
                external_call_id=f"session:{input.session_id}:turn:{turn_id}:initial",
                metadata={"session_id": input.session_id, "turn_id": turn_id},
            )
            self._record_model_response(
                session_id=input.session_id,
                turn_id=turn_id,
                response=initial_response,
            )

            tool_calls = self._resolve_tool_calls(
                response=initial_response,
                strategy=strategy,
            )
            if not tool_calls:
                final_output = Message(role="assistant", content=initial_response.text)
                self._append_event(
                    session_id=input.session_id,
                    turn_id=turn_id,
                    event_type=SessionEventType.model_completed,
                    payload=SessionMessagePayload(message=final_output),
                )
                self._append_event(
                    session_id=input.session_id,
                    turn_id=turn_id,
                    event_type=SessionEventType.message,
                    payload=SessionMessagePayload(message=final_output),
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

            tool_processing = self._handle_tool_calls(
                state=state,
                input=input,
                session_metadata=session_metadata,
                turn_id=turn_id,
                strategy=strategy,
                supports_native_tools=adapter.capabilities.supports_native_tools,
                resolved_reasoning=resolved_reasoning,
                conversation=conversation,
                tool_calls=tool_calls,
                initial_response=initial_response,
            )
            if tool_processing.waiting_for_tools:
                return SessionStepResult(
                    session_id=input.session_id,
                    turn_id=turn_id,
                    status=SessionTurnStatus.active,
                    version=new_version,
                    output=tool_processing.final_output,
                    tool_calls=tool_processing.tool_calls,
                )

            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=SessionEventType.model_completed,
                payload=SessionMessagePayload(message=tool_processing.final_output),
            )
            self._append_event(
                session_id=input.session_id,
                turn_id=turn_id,
                event_type=SessionEventType.message,
                payload=SessionMessagePayload(message=tool_processing.final_output),
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
                output=tool_processing.final_output,
                tool_calls=tool_processing.tool_calls,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_step_failure(
                session_id=input.session_id,
                turn_id=turn_id,
                exc=exc,
            )
            raise
