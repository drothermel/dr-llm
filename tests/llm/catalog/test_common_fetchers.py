from __future__ import annotations

from typing import Any, cast

import pytest

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.catalog.fetchers.common import (
    fetch_models_with_template,
    require_api_key,
)
from dr_llm.llm.catalog.fetchers.static import fetch_static_headless_models
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.providers.config import ProviderConfig
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from tests.conftest import make_response


class _UnsupportedProvider(Provider):
    def __init__(self) -> None:
        self._config = ProviderConfig(name="unsupported-headless")

    def generate(self, request: LlmRequest) -> LlmResponse:
        return make_response(provider=request.provider, model=request.model)


def test_require_api_key_uses_env_fallback_for_whitespace_explicit_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "env-secret")

    assert (
        require_api_key(
            api_key="   ",
            env_var="TEST_API_KEY",
            label="test provider",
        )
        == "env-secret"
    )


def test_require_api_key_raises_when_explicit_and_env_values_are_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "   ")

    with pytest.raises(
        ProviderSemanticError,
        match="Missing test provider API key for catalog sync. Set TEST_API_KEY",
    ):
        require_api_key(
            api_key="   ",
            env_var="TEST_API_KEY",
            label="test provider",
        )


def test_fetch_models_with_template_raises_when_items_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dr_llm.llm.catalog.fetchers.common.get_json",
        lambda *, url, headers=None, timeout_seconds=30.0: {"unexpected": []},
    )

    with pytest.raises(
        ValueError,
        match=r"missing items_key='models'.*payload=\{'unexpected': \[\]\}",
    ):
        fetch_models_with_template(
            url="https://example.com/models",
            headers=None,
            items_key="models",
            item_processor=lambda item, fetched_at: None,
        )


def test_fetch_models_with_template_raises_when_items_key_is_not_a_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dr_llm.llm.catalog.fetchers.common.get_json",
        lambda *, url, headers=None, timeout_seconds=30.0: {"models": {}},
    )

    with pytest.raises(
        ValueError,
        match=r"non-list items_key='models'.*payload=\{'models': \{\}\}",
    ):
        fetch_models_with_template(
            url="https://example.com/models",
            headers=None,
            items_key="models",
            item_processor=lambda item, fetched_at: None,
        )


def test_fetch_static_headless_models_raises_for_unknown_provider_type() -> None:
    provider = cast(Any, _UnsupportedProvider())

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported static headless provider for catalog fetch: "
            "type=_UnsupportedProvider name='unsupported-headless'"
        ),
    ):
        fetch_static_headless_models(provider)
