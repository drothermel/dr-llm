from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from llm_pool.storage.repository import PostgresRepository
from llm_pool.tools.executor import ToolExecutor
from llm_pool.types import Message, ToolCallStatus, ToolInvocation


class ToolWorkerEventType(StrEnum):
    tool_succeeded = "tool_succeeded"
    tool_result_message = "tool_result_message"
    tool_failed = "tool_failed"
    tool_retry_scheduled = "tool_retry_scheduled"


class ToolWorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: str | None = None
    lease_seconds: int = Field(default=60, gt=0)
    batch_size: int = Field(default=8, gt=0)
    idle_sleep_seconds: float = Field(default=0.5, ge=0)
    max_loops: int | None = Field(default=None, ge=0)
    max_attempts_before_dead_letter: int = Field(default=3, ge=0)


class ToolWorkerStats(BaseModel):
    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    dead_lettered: int = 0


class ToolSucceededPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    result: dict[str, Any] | None = None


class ToolResultMessagePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: Message


class ToolFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    error: dict[str, Any] | None = None
    terminal: bool


class ToolRetryScheduledPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    error: dict[str, Any] | None = None
    attempt_count: int


class ToolDeadLetterPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: str
    status: ToolCallStatus
    error: dict[str, Any] | None = None


def _payload_dict(payload: BaseModel) -> dict[str, Any]:
    return payload.model_dump(
        mode="json", exclude_none=True, exclude_computed_fields=True
    )


def _append_event(
    *,
    repository: PostgresRepository,
    session_id: str,
    turn_id: str | None,
    event_type: ToolWorkerEventType,
    payload: BaseModel,
) -> None:
    repository.append_session_event(
        session_id=session_id,
        turn_id=turn_id,
        event_type=event_type.value,
        payload=_payload_dict(payload),
    )


def _serialize_tool_result(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=True, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(result)


def run_tool_worker(
    *,
    repository: PostgresRepository,
    executor: ToolExecutor,
    worker_id: str | None = None,
    lease_seconds: int = 60,
    batch_size: int = 8,
    idle_sleep_seconds: float = 0.5,
    max_loops: int | None = None,
    max_attempts_before_dead_letter: int = 3,
) -> dict[str, Any]:
    config = ToolWorkerConfig(
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        batch_size=batch_size,
        idle_sleep_seconds=idle_sleep_seconds,
        max_loops=max_loops,
        max_attempts_before_dead_letter=max_attempts_before_dead_letter,
    )
    wid = config.worker_id or f"tool-worker-{uuid4().hex[:8]}"
    loops = 0
    stats = ToolWorkerStats()
    while config.max_loops is None or loops < config.max_loops:
        loops += 1
        claimed = repository.claim_tool_calls(
            worker_id=wid,
            limit=config.batch_size,
            lease_seconds=config.lease_seconds,
        )
        if not claimed:
            time.sleep(config.idle_sleep_seconds)
            continue

        stats.claimed += len(claimed)
        for call in claimed:
            try:
                invocation = ToolInvocation(
                    tool_call_id=call.tool_call_id,
                    name=call.tool_name,
                    arguments=call.args,
                    session_id=call.session_id,
                    turn_id=call.turn_id,
                )
                result = executor.invoke(invocation)
                if result.ok:
                    repository.complete_tool_call(result=result)
                    _append_event(
                        repository=repository,
                        session_id=call.session_id,
                        turn_id=call.turn_id,
                        event_type=ToolWorkerEventType.tool_succeeded,
                        payload=ToolSucceededPayload(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            result=result.result,
                        ),
                    )
                    tool_message = Message(
                        role="tool",
                        name=call.tool_name,
                        content=_serialize_tool_result(result.result or {}),
                        tool_call_id=call.tool_call_id,
                    )
                    _append_event(
                        repository=repository,
                        session_id=call.session_id,
                        turn_id=call.turn_id,
                        event_type=ToolWorkerEventType.tool_result_message,
                        payload=ToolResultMessagePayload(message=tool_message),
                    )
                    stats.succeeded += 1
                    continue

                stats.failed += 1
                error_payload = (
                    result.error.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                    if result.error is not None
                    else None
                )
                if call.attempt_count >= config.max_attempts_before_dead_letter:
                    repository.dead_letter_tool_call(
                        tool_call_id=call.tool_call_id,
                        reason=result.error.message
                        if result.error is not None
                        else "unknown tool error",
                        payload=_payload_dict(
                            ToolDeadLetterPayload(
                                worker_id=wid,
                                status=ToolCallStatus.failed,
                                error=error_payload,
                            )
                        ),
                    )
                    _append_event(
                        repository=repository,
                        session_id=call.session_id,
                        turn_id=call.turn_id,
                        event_type=ToolWorkerEventType.tool_failed,
                        payload=ToolFailedPayload(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            error=error_payload,
                            terminal=True,
                        ),
                    )
                    stats.dead_lettered += 1
                else:
                    release_error_text = (
                        json.dumps(error_payload, ensure_ascii=True, sort_keys=True)
                        if error_payload is not None
                        else None
                    )
                    repository.release_tool_claim(
                        tool_call_id=call.tool_call_id,
                        worker_id=wid,
                        error_text=release_error_text,
                    )
                    _append_event(
                        repository=repository,
                        session_id=call.session_id,
                        turn_id=call.turn_id,
                        event_type=ToolWorkerEventType.tool_retry_scheduled,
                        payload=ToolRetryScheduledPayload(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            error=error_payload,
                            attempt_count=call.attempt_count,
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                stats.failed += 1
                exc_payload = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                if call.attempt_count >= config.max_attempts_before_dead_letter:
                    repository.dead_letter_tool_call(
                        tool_call_id=call.tool_call_id,
                        reason=f"worker_exception:{type(exc).__name__}",
                        payload=_payload_dict(
                            ToolDeadLetterPayload(
                                worker_id=wid,
                                status=ToolCallStatus.failed,
                                error=exc_payload,
                            )
                        ),
                    )
                    _append_event(
                        repository=repository,
                        session_id=call.session_id,
                        turn_id=call.turn_id,
                        event_type=ToolWorkerEventType.tool_failed,
                        payload=ToolFailedPayload(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            error=exc_payload,
                            terminal=True,
                        ),
                    )
                    stats.dead_lettered += 1
                else:
                    repository.release_tool_claim(
                        tool_call_id=call.tool_call_id,
                        worker_id=wid,
                        error_text=str(exc),
                    )
                    _append_event(
                        repository=repository,
                        session_id=call.session_id,
                        turn_id=call.turn_id,
                        event_type=ToolWorkerEventType.tool_retry_scheduled,
                        payload=ToolRetryScheduledPayload(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            error=exc_payload,
                            attempt_count=call.attempt_count,
                        ),
                    )
        if config.max_loops is None:
            continue
    return stats.model_dump(mode="json", exclude_computed_fields=True)
