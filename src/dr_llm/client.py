from __future__ import annotations

from uuid import uuid4
from typing import Any

from dr_llm.catalog.models import (
    ModelCatalogEntry,
    ModelCatalogQuery,
    ModelCatalogSyncResult,
)
from dr_llm.catalog.service import ModelCatalogService
from dr_llm.generation.models import LlmRequest, LlmResponse
from dr_llm.logging import emit_generation_event, generation_log_context
from dr_llm.providers import build_default_registry
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.storage.repository import PostgresRepository


class LlmClient:
    """Unified LLM client for API and headless adapters."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry | None = None,
        repository: PostgresRepository | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.repository = repository
        self._catalog = ModelCatalogService(
            registry=self.registry,
            repository=self.repository,
        )

    def get_adapter(self, provider_name: str) -> ProviderAdapter:
        return self.registry.get(provider_name)

    def provider_capabilities(self, provider_name: str) -> dict[str, bool]:
        adapter = self.get_adapter(provider_name)
        return {
            "supports_structured_output": adapter.config.supports_structured_output,
        }

    def known_providers(self) -> list[str]:
        return sorted(self.registry.names())

    def sync_models(self, provider: str | None = None) -> dict[str, int]:
        return self._catalog.sync_models(provider=provider)

    def sync_models_detailed(
        self, provider: str | None = None
    ) -> list[ModelCatalogSyncResult]:
        return self._catalog.sync_models_detailed(provider=provider)

    def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        return self._catalog.list_models(query)

    def count_models(self, query: ModelCatalogQuery) -> int:
        return self._catalog.count_models(query)

    def show_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        return self._catalog.show_model(provider=provider, model=model)

    def query(
        self,
        request: LlmRequest,
        *,
        run_id: str | None = None,
        external_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LlmResponse:
        adapter = self.get_adapter(request.provider)
        call_id = external_call_id or uuid4().hex
        log_context = {
            "call_id": call_id,
            "run_id": run_id,
            "provider": request.provider,
            "model": request.model,
            "mode": adapter.mode,
        }
        with generation_log_context(log_context):
            emit_generation_event(
                event_type="llm_call.started",
                stage="client.before_adapter",
                payload={
                    "request": request.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                },
            )
            try:
                response = adapter.generate(request)
            except Exception as exc:  # noqa: BLE001
                emit_generation_event(
                    event_type="llm_call.failed",
                    stage="client.adapter_exception",
                    payload={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                if self.repository is not None:
                    try:
                        self.repository.record_call(
                            request=request,
                            response=None,
                            run_id=run_id,
                            status="failed",
                            mode=adapter.mode,
                            error_text=str(exc),
                            external_call_id=external_call_id,
                            metadata=metadata,
                            call_id=call_id,
                        )
                    except Exception as record_exc:  # noqa: BLE001
                        emit_generation_event(
                            event_type="llm_call.db_record_failed",
                            stage="client.record_failure",
                            payload={
                                "error_type": type(record_exc).__name__,
                                "message": str(record_exc),
                            },
                        )
                raise

            emit_generation_event(
                event_type="llm_call.succeeded",
                stage="client.after_adapter",
                payload={
                    "response": response.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                },
            )
            if self.repository is not None:
                try:
                    self.repository.record_call(
                        request=request,
                        response=response,
                        run_id=run_id,
                        status="success",
                        mode=adapter.mode,
                        external_call_id=external_call_id,
                        metadata=metadata,
                        call_id=call_id,
                    )
                except Exception as record_exc:  # noqa: BLE001
                    emit_generation_event(
                        event_type="llm_call.db_record_failed",
                        stage="client.record_success",
                        payload={
                            "error_type": type(record_exc).__name__,
                            "message": str(record_exc),
                        },
                    )
        return response
