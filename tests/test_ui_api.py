from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.config import ProviderConfig
from dr_llm.llm.providers.registry import ProviderRegistry
from tests.conftest import FakeProvider
from ui.api import main as ui_api


def test_providers_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeProvider(
        "fake-provider",
        config=ProviderConfig(name="fake-provider", supports_structured_output=True),
    )
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


def test_openrouter_models_endpoint_applies_policy_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeProvider("openrouter", config=ProviderConfig(name="openrouter"))
    registry = ProviderRegistry()
    registry.register(adapter)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)
    monkeypatch.setattr(
        ui_api,
        "fetch_models_for_provider",
        lambda _provider: (
            [
                ModelCatalogEntry(
                    provider="openrouter",
                    model="deepseek/deepseek-chat-v3.1",
                    supports_reasoning=False,
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider="openrouter",
                    model="deepseek/deepseek-chat",
                    supports_reasoning=True,
                    source_quality="live",
                ),
                ModelCatalogEntry(
                    provider="openrouter",
                    model="unknown/model",
                    source_quality="live",
                ),
            ],
            {},
        ),
    )

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers/openrouter/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "live"
    assert [model["model"] for model in payload["models"]] == [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat",
    ]
    assert payload["models"][0]["supports_reasoning"] is True
    assert payload["models"][1]["supports_reasoning"] is False


def test_openrouter_static_models_use_curated_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeProvider(
        "openrouter",
        config=ProviderConfig(
            name="openrouter", required_env_vars=["MISSING_TEST_ENV"]
        ),
    )
    registry = ProviderRegistry()
    registry.register(adapter)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers/openrouter/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "static"
    assert any(model["model"] == "openai/gpt-oss-20b" for model in payload["models"])
    assert any(
        model["model"] == "deepseek/deepseek-chat-v3.1" for model in payload["models"]
    )
