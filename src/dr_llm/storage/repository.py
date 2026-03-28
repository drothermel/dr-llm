from __future__ import annotations

from typing import Any

from dr_llm.storage._catalog_store import CatalogStore
from dr_llm.storage._runs_calls_store import RunsCallsStore
from dr_llm.storage._runtime import StorageConfig, StorageRuntime
from dr_llm.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    ModelCatalogEntry,
    ModelCatalogQuery,
    RecordedCall,
    RunStatus,
)


class PostgresRepository:
    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self._runtime = StorageRuntime(self.config)
        self._runs_calls = RunsCallsStore(self._runtime)
        self._catalog = CatalogStore(self._runtime)

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
