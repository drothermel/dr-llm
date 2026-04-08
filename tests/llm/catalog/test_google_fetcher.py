from __future__ import annotations

from typing import Any

from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.google.provider import GoogleProvider


def test_google_catalog_fetch_passes_api_key_via_header(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_get_json(
        *,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout_seconds"] = timeout_seconds
        return {"models": []}

    monkeypatch.setattr("dr_llm.llm.catalog.fetchers.google.get_json", fake_get_json)
    provider = GoogleProvider(
        config=APIProviderConfig(
            name="google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GOOGLE_API_KEY",
            api_key="secret",
        )
    )

    entries, payload = fetch_google_models(provider)

    assert entries == []
    assert payload == {"models": []}
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models"
    assert captured["headers"] == {"x-goog-api-key": "secret"}
    assert "?key=" not in captured["url"]
