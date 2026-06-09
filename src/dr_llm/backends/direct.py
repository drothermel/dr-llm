"""Direct provider backend without pool caching."""

from __future__ import annotations

import asyncio

from dr_llm.backends.converters import (
    capabilities_from_controls,
    llm_response_to_backend_response,
)
from dr_llm.backends.fingerprint import fingerprint_request
from dr_llm.backends.models import (
    BackendCapabilities,
    BackendRequest,
    BackendResponse,
)
from dr_llm.backends.validation import validate_v1_request
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.default_registry import build_default_registry


class DirectBackend:
    """Provider-only backend that calls orchestrators directly."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()

    def complete(self, request: BackendRequest) -> BackendResponse:
        validate_v1_request(request)
        orchestrator = self._registry.get(request.provider)
        llm_response = orchestrator.generate(request.to_llm_request())
        return llm_response_to_backend_response(
            llm_response,
            source="direct",
            fingerprint=fingerprint_request(request),
        )

    async def acomplete(self, request: BackendRequest) -> BackendResponse:
        return await asyncio.to_thread(self.complete, request)

    def capabilities(self, request: BackendRequest) -> BackendCapabilities:
        orchestrator = self._registry.get(request.provider)
        return capabilities_from_controls(orchestrator.controls(request.model))
