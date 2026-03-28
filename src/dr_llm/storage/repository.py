from __future__ import annotations

from typing import Any

from dr_llm.storage._catalog_store import CatalogStore
from dr_llm.storage._runs_calls_store import RunsCallsStore
from dr_llm.storage._runtime import StorageConfig, StorageRuntime
from dr_llm.storage._sessions_store import SessionsStore
from dr_llm.storage._tools_store import ToolsStore
from dr_llm.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    ModelCatalogEntry,
    ModelCatalogQuery,
    RecordedCall,
    RunStatus,
    SessionEvent,
    SessionHandle,
    SessionState,
    SessionStatus,
    SessionTurnStatus,
    ToolCallRecord,
    ToolPolicy,
    ToolResult,
)


class PostgresRepository:
    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self._runtime = StorageRuntime(self.config)
        self._runs_calls = RunsCallsStore(self._runtime)
        self._catalog = CatalogStore(self._runtime)
        self._sessions = SessionsStore(self._runtime)
        self._tools = ToolsStore(self._runtime)

    def close(self) -> None:
        self._runtime.close()

    def init_schema(self) -> None:
        self._runtime.init_schema()

    def initialize(self) -> None:
        self._runtime.initialize()

    def start_run(
        self,
        *,
        run_type: str = "generic",
        status: RunStatus = RunStatus.running,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        return self._runs_calls.start_run(
            run_type=run_type,
            status=status,
            metadata=metadata,
            run_id=run_id,
        )

    def upsert_run_parameters(self, *, run_id: str, parameters: dict[str, Any]) -> int:
        return self._runs_calls.upsert_run_parameters(
            run_id=run_id,
            parameters=parameters,
        )

    def finish_run(
        self,
        *,
        run_id: str,
        status: RunStatus,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._runs_calls.finish_run(run_id=run_id, status=status, metadata=metadata)

    def record_call(
        self,
        *,
        request: LlmRequest,
        response: LlmResponse | None = None,
        run_id: str | None = None,
        status: str | None = None,
        mode: CallMode | str | None = None,
        error_text: str | None = None,
        external_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> str:
        return self._runs_calls.record_call(
            request=request,
            response=response,
            run_id=run_id,
            status=status,
            mode=mode,
            error_text=error_text,
            external_call_id=external_call_id,
            metadata=metadata,
            call_id=call_id,
        )

    def record_calls_batch(
        self,
        *,
        requests: list[LlmRequest],
        responses: list[LlmResponse | None],
        run_id: str | None = None,
    ) -> list[str]:
        return self._runs_calls.record_calls_batch(
            requests=requests,
            responses=responses,
            run_id=run_id,
        )

    def list_calls(
        self,
        *,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RecordedCall]:
        return self._runs_calls.list_calls(run_id=run_id, limit=limit, offset=offset)

    def record_artifact(
        self,
        *,
        run_id: str | None,
        artifact_type: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._runs_calls.record_artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            metadata=metadata,
        )

    def record_model_catalog_snapshot(
        self,
        *,
        provider: str,
        status: str,
        raw_payload: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> str:
        return self._catalog.record_catalog_snapshot(
            provider=provider,
            status=status,
            raw_payload=raw_payload,
            error_text=error_text,
        )

    def replace_provider_models(
        self,
        *,
        provider: str,
        entries: list[ModelCatalogEntry],
    ) -> int:
        return self._catalog.replace_provider_models(provider=provider, entries=entries)

    def list_models(self, *, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        return self._catalog.list_models(query=query)

    def count_models(self, *, query: ModelCatalogQuery) -> int:
        return self._catalog.count_models(query=query)

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        return self._catalog.get_model(provider=provider, model=model)

    def start_session(
        self,
        *,
        strategy_mode: ToolPolicy = ToolPolicy.native_preferred,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionHandle:
        return self._sessions.start_session(
            strategy_mode=strategy_mode,
            metadata=metadata,
            session_id=session_id,
        )

    def get_session(self, *, session_id: str) -> SessionState:
        return self._sessions.get_session(session_id=session_id)

    def advance_session_version(self, *, session_id: str, expected_version: int) -> int:
        return self._sessions.advance_session_version(
            session_id=session_id,
            expected_version=expected_version,
        )

    def update_session_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        last_error_text: str | None = None,
    ) -> None:
        self._sessions.update_session_status(
            session_id=session_id,
            status=status,
            last_error_text=last_error_text,
        )

    def create_session_turn(
        self,
        *,
        session_id: str,
        status: SessionTurnStatus = SessionTurnStatus.active,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        return self._sessions.create_session_turn(
            session_id=session_id,
            status=status,
            metadata=metadata,
        )

    def complete_session_turn(
        self,
        *,
        turn_id: str,
        status: SessionTurnStatus,
    ) -> None:
        self._sessions.complete_session_turn(turn_id=turn_id, status=status)

    def append_session_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        return self._sessions.append_session_event(
            session_id=session_id,
            event_type=event_type,
            payload=payload,
            turn_id=turn_id,
            event_id=event_id,
        )

    def load_session_events(self, *, session_id: str) -> list[SessionEvent]:
        return self._sessions.load_session_events(session_id=session_id)

    def replay_session_messages(self, *, session_id: str) -> list[dict[str, Any]]:
        return self._sessions.replay_session_messages(session_id=session_id)

    def enqueue_tool_call(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        idempotency_key: str,
        turn_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        return self._tools.enqueue_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            args=args,
            idempotency_key=idempotency_key,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
        )

    def claim_tool_calls(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
    ) -> list[ToolCallRecord]:
        return self._tools.claim_tool_calls(
            worker_id=worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )

    def renew_tool_lease(
        self,
        *,
        tool_call_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        return self._tools.renew_tool_lease(
            tool_call_id=tool_call_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )

    def release_tool_claim(
        self,
        *,
        tool_call_id: str,
        worker_id: str,
        error_text: str | None = None,
    ) -> None:
        self._tools.release_tool_claim(
            tool_call_id=tool_call_id,
            worker_id=worker_id,
            error_text=error_text,
        )

    def complete_tool_call(self, *, result: ToolResult) -> None:
        self._tools.complete_tool_call(result=result)

    def dead_letter_tool_call(
        self,
        *,
        tool_call_id: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        return self._tools.dead_letter_tool_call(
            tool_call_id=tool_call_id,
            reason=reason,
            payload=payload,
        )
