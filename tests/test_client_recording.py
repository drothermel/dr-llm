from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from dr_llm.client import LlmClient
from dr_llm.providers.base import (
    ProviderAdapter,
    ProviderCapabilities,
    ProviderRuntimeRequirements,
)
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.generation.models import LlmRequest, Message


class CapturingRepository(BaseModel):
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FailingHeadlessAdapter(ProviderAdapter):
    name = "failing-headless"
    mode = "headless"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_structured_output=True)

    @property
    def runtime_requirements(self) -> ProviderRuntimeRequirements:
        return ProviderRuntimeRequirements()

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
