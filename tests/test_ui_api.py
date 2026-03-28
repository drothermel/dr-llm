from __future__ import annotations

from fastapi.testclient import TestClient

from dr_llm.providers.base import (
    ProviderAdapter,
    ProviderCapabilities,
    ProviderRuntimeRequirements,
)
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.types import CallMode, LlmRequest, LlmResponse, TokenUsage
from ui.api import main as ui_api


class _UiFakeAdapter(ProviderAdapter):
    name = "fake-provider"
    mode = "api"

    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_structured_output=True)

    @property
    def runtime_requirements(self) -> ProviderRuntimeRequirements:
        return ProviderRuntimeRequirements()

    def generate(self, request: LlmRequest) -> LlmResponse:  # noqa: ARG002
        return LlmResponse(
            text="ok",
            usage=TokenUsage(),
            provider="fake-provider",
            model="fake-model",
            mode=CallMode.api,
        )

    def close(self) -> None:
        self.close_calls += 1


def test_ui_api_builds_registry_on_startup_and_closes_on_shutdown(monkeypatch) -> None:
    adapter = _UiFakeAdapter()
    registry = ProviderRegistry()
    registry.register(adapter)

    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)
    monkeypatch.setattr(ui_api, "_registry", None)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers")

    assert response.status_code == 200
    assert response.json() == [
        {
            "provider": "fake-provider",
            "available": True,
            "missing_env_vars": [],
            "missing_executables": [],
            "supports_structured_output": True,
        }
    ]
    assert adapter.close_calls == 1
    assert ui_api._registry is None
