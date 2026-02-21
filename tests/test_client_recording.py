from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from llm_pool.client import LlmClient
from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.registry import ProviderRegistry
from llm_pool.types import LlmRequest, Message


class CapturingRepository(BaseModel):
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FailingHeadlessAdapter(ProviderAdapter):
    name = "failing-headless"
    mode = "headless"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_native_tools=False, supports_structured_output=True
        )

    def generate(self, request: Any) -> Any:
        _ = request
        raise RuntimeError("boom")


def test_failed_query_records_adapter_mode() -> None:
    registry = ProviderRegistry()
    registry.register(FailingHeadlessAdapter())
    repo = CapturingRepository(calls=[])
    # CapturingRepository is a duck-typed test double for LlmClient(repository=...).
    client = LlmClient(registry=registry, repository=cast(Any, repo))

    request = LlmRequest(
        provider="failing-headless",
        model="dummy",
        messages=[Message(role="user", content="hi")],
    )

    with pytest.raises(RuntimeError):
        client.query(request)

    assert len(repo.calls) == 1
    assert repo.calls[0]["mode"] == "headless"
