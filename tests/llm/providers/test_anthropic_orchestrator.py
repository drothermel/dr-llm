from __future__ import annotations

from collections.abc import Generator

import pytest

from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.reasoning import AnthropicReasoning
from dr_llm.llm.providers.impls.anthropic.orchestrator import (
    AnthropicOrchestrator,
)
from dr_llm.llm.providers.impls.anthropic.provider import AnthropicProvider


@pytest.fixture
def orchestrator() -> Generator[AnthropicOrchestrator]:
    provider = AnthropicProvider()
    orch = AnthropicOrchestrator(provider)
    yield orch
    provider.close()


class TestModelCapabilities:
    def test_opus_46_has_effort_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-opus-4-6")
        assert EffortSpec.MAX in caps.supported_effort_levels
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT

    def test_sonnet_46_has_effort_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-sonnet-4-6")
        assert EffortSpec.MAX not in caps.supported_effort_levels
        assert EffortSpec.HIGH in caps.supported_effort_levels

    def test_sonnet_46_snapshot_uses_same_family_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-sonnet-4-6-20261201")
        assert EffortSpec.HIGH in caps.supported_effort_levels
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT

    def test_opus_45_has_reasoning_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-opus-4-5-20251101")
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET
        assert caps.reasoning.min_budget_tokens == 1024

    def test_opus_45_snapshot_uses_same_effort_and_budget_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-opus-4-5-20261201")
        assert EffortSpec.HIGH in caps.supported_effort_levels
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET

    def test_sonnet_45_has_reasoning_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-sonnet-4-5-20241022")
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_BUDGET
        assert caps.supported_effort_levels == ()

    def test_haiku_45_has_reasoning_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-haiku-4-5-20241022")
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_BUDGET

    def test_unknown_model_returns_none_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-2.1")
        assert caps.reasoning.mode == ReasoningMode.UNSUPPORTED
        assert caps.supported_effort_levels == ()


class TestValidateRequest:
    def test_valid_reasoning_returns_no_warnings(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        from dr_llm.llm import AnthropicReasoning, ThinkingLevel
        from tests.conftest import make_request

        request = make_request(
            provider=ProviderName.ANTHROPIC,
            model="claude-sonnet-4-5-20241022",
            max_tokens=1024,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET, budget_tokens=2048
            ),
        )
        warnings = orchestrator.validate_request(request)
        assert warnings == []

    def test_adaptive_model_returns_no_warnings(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        from dr_llm.llm import AnthropicReasoning, ThinkingLevel
        from tests.conftest import make_request

        request = make_request(
            provider=ProviderName.ANTHROPIC,
            model="claude-sonnet-4-6",
            max_tokens=1024,
            effort=EffortSpec.HIGH,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE
            ),
        )
        warnings = orchestrator.validate_request(request)
        assert warnings == []


class TestProperties:
    def test_validate_request_uses_anthropic_reasoning_validator(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        from tests.conftest import make_request

        request = make_request(
            provider=ProviderName.ANTHROPIC,
            model="claude-sonnet-4-5-20241022",
            max_tokens=1024,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE
            ),
        )

        with pytest.raises(ValueError, match="adaptive thinking"):
            orchestrator.validate_request(request)
