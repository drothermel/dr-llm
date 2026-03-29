from __future__ import annotations

from fastapi.testclient import TestClient

from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.provider_config import ProviderConfig
from dr_llm.generation.models import CallMode, LlmRequest, LlmResponse, TokenUsage
from dr_llm.providers.registry import ProviderRegistry
from ui.api import main as ui_api


class _UiFakeAdapter(ProviderAdapter):
    def __init__(self) -> None:
        self._config = ProviderConfig(
            name="fake-provider",
            supports_structured_output=True,
        )
        self.close_calls = 0

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
    assert getattr(ui_api.app.state, "registry", None) is None
