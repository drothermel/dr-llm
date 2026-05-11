from __future__ import annotations

from dr_llm.llm import ControlMode, ProviderName
import pytest
from fastapi.testclient import TestClient

from dr_llm.llm import ProviderConfig, ProviderRegistry
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.impls.openrouter.orchestrator import (
    OpenRouterOrchestrator,
)
from dr_llm.llm.providers.impls.openrouter.provider import OpenRouterProvider
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from tests.conftest import FakeOrchestrator
from ui.api import main as ui_api


def test_providers_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = FakeOrchestrator(
        "fake-provider",
        config=ProviderConfig(
            name="fake-provider", supports_structured_output=True
        ),
    )
    registry = ProviderRegistry()
    registry.register(orchestrator)
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
    assert orchestrator.close_calls == 1
    assert getattr(ui_api.app.state, "registry", None) is None


def test_models_endpoint_uses_orchestrator_catalog_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="deepseek/deepseek-chat-v3.1",
            control_mode=ControlMode.UNSUPPORTED,
            source_quality="live",
        ),
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="deepseek/deepseek-chat",
            control_mode=ControlMode.OPENROUTER_TOGGLE,
            source_quality="live",
        ),
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="unknown/model",
            source_quality="live",
        ),
    ]
    orchestrator = FakeOrchestrator(
        ProviderName.OPENROUTER,
        config=ProviderConfig(name=ProviderName.OPENROUTER),
        fetch_models_fn=lambda: (entries, {}),
    )
    registry = ProviderRegistry()
    registry.register(orchestrator)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers/openrouter/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "live"
    assert [model["model"] for model in payload["models"]] == [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat",
        "unknown/model",
    ]
    assert payload["models"][0]["control_mode"] == "unsupported"
    assert payload["models"][1]["control_mode"] == "openrouter_toggle"


def test_openrouter_static_models_come_from_orchestrator_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = OpenRouterOrchestrator(
        OpenRouterProvider(
            config=OpenAICompatConfig(
                name=ProviderName.OPENROUTER,
                base_url="https://openrouter.ai/api/v1",
                api_key_env="MISSING_TEST_ENV",
                required_env_vars=["MISSING_TEST_ENV"],
            )
        ),
    )
    registry = ProviderRegistry()
    registry.register(orchestrator)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers/openrouter/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "static"
    assert any(
        model["model"] == "openai/gpt-oss-20b" for model in payload["models"]
    )
    assert any(
        model["model"] == "deepseek/deepseek-chat-v3.1"
        for model in payload["models"]
    )
