from __future__ import annotations

from typing import Any

from llm_pool.providers import build_default_registry
from llm_pool.providers.base import ProviderAdapter
from llm_pool.providers.registry import ProviderRegistry
from llm_pool.storage.repository import PostgresRepository
from llm_pool.types import LlmRequest, LlmResponse


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

    def get_adapter(self, provider_name: str) -> ProviderAdapter:
        return self.registry.get(provider_name)

    def provider_capabilities(self, provider_name: str) -> dict[str, bool]:
        adapter = self.get_adapter(provider_name)
        caps = adapter.capabilities
        return {
            "supports_native_tools": caps.supports_native_tools,
            "supports_structured_output": caps.supports_structured_output,
        }

    def known_providers(self) -> list[str]:
        return sorted(self.registry.names())

    def query(
        self,
        request: LlmRequest,
        *,
        run_id: str | None = None,
        external_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LlmResponse:
        adapter = self.get_adapter(request.provider)
        try:
            response = adapter.generate(request)
        except Exception as exc:  # noqa: BLE001
            if self.repository is not None:
                self.repository.record_call(
                    request=request,
                    response=None,
                    run_id=run_id,
                    status="failed",
                    error_text=str(exc),
                    external_call_id=external_call_id,
                    metadata=metadata,
                )
            raise

        if self.repository is not None:
            self.repository.record_call(
                request=request,
                response=response,
                run_id=run_id,
                status="success",
                external_call_id=external_call_id,
                metadata=metadata,
            )
        return response
