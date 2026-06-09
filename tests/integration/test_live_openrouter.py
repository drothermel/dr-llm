from __future__ import annotations

from os import getenv

import pytest

from dr_llm.backends import BackendRequest, DirectBackend
from dr_llm.llm import (
    CallMode,
    Message,
    OpenRouterEffortLevel,
    ProviderName,
    build_default_registry,
)
from dr_llm.llm.providers.concepts.reasoning import OpenRouterReasoning

pytestmark = pytest.mark.integration


def test_live_openrouter_gpt5_nano_direct_backend_smoke() -> None:
    if not getenv("OPENROUTER_API_KEY"):
        pytest.skip("Set OPENROUTER_API_KEY to run live OpenRouter smoke test")

    registry = build_default_registry()
    try:
        backend = DirectBackend(registry)
        response = backend.complete(
            BackendRequest(
                provider=ProviderName.OPENROUTER,
                model="openai/gpt-5-nano",
                mode=CallMode.api,
                messages=[
                    Message(
                        role="user",
                        content="Return exactly: dr-llm live ok",
                    )
                ],
                max_tokens=256,
                reasoning=OpenRouterReasoning(
                    effort=OpenRouterEffortLevel.LOW
                ),
            )
        )
    finally:
        registry.close()

    assert response.text.strip()
    assert response.provider == ProviderName.OPENROUTER
    assert response.model == "openai/gpt-5-nano"
    assert response.source == "direct"
    assert response.request_fingerprint is not None
