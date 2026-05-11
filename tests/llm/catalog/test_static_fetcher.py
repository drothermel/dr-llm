from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import (
    build_static_catalog_entries,
    display_str,
)
from dr_llm.llm.providers.core.base import ProviderTransport
from dr_llm.llm.providers.core.config import ProviderConfig
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse


class StaticCatalogTestProvider(ProviderTransport):
    def __init__(self) -> None:
        self._config = ProviderConfig(name="test")

    def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError


def test_display_str_replaces_dashes_with_title_case() -> None:
    assert display_str("gpt-5.4-mini") == "Gpt 5.4 Mini"


def test_build_static_catalog_entries_derives_display_name() -> None:
    entries, raw_payload = build_static_catalog_entries(
        provider=StaticCatalogTestProvider(),
        models=["claude-sonnet-4-6"],
        docs_url="https://example.com/models",
        supports_vision=True,
        capabilities_fn=lambda model: None,
    )

    assert raw_payload == {
        "source": "static",
        "docs_url": "https://example.com/models",
    }
    assert entries[0].model == "claude-sonnet-4-6"
    assert entries[0].display_name == "Claude Sonnet 4 6"
