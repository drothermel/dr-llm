from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dr_llm.providers.provider_config import ProviderConfig
from dr_llm.providers.registry import ProviderRegistry
from tests.conftest import FakeAdapter
from ui.api import main as ui_api


def test_providers_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(
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
