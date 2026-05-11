from __future__ import annotations

from collections.abc import Generator

import pytest

from dr_llm.llm.names import (
    ControlStrategy,
    EffortSpec,
    ProviderName,
    ReasoningMode,
)
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
        assert caps.control_strategy == ControlStrategy.EFFORT
        assert EffortSpec.MAX in caps.supported_effort_levels
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT

    def test_sonnet_46_has_effort_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-sonnet-4-6")
        assert caps.control_strategy == ControlStrategy.EFFORT
        assert EffortSpec.MAX not in caps.supported_effort_levels
        assert EffortSpec.HIGH in caps.supported_effort_levels

    def test_opus_45_has_reasoning_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-opus-4-5-20251101")
        assert caps.control_strategy == ControlStrategy.EFFORT
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_EFFORT_AND_BUDGET
        assert caps.reasoning.min_budget_tokens == 1024

    def test_sonnet_45_has_reasoning_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-sonnet-4-5-20241022")
        assert caps.control_strategy == ControlStrategy.REASONING
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_BUDGET
        assert caps.supported_effort_levels == ()

    def test_haiku_45_has_reasoning_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-haiku-4-5-20241022")
        assert caps.control_strategy == ControlStrategy.REASONING
        assert caps.reasoning.mode == ReasoningMode.ANTHROPIC_BUDGET

    def test_unknown_model_returns_none_strategy(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        caps = orchestrator.model_capabilities("claude-2.1")
        assert caps.control_strategy == ControlStrategy.NONE
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
    def test_name_is_anthropic(
        self, orchestrator: AnthropicOrchestrator
    ) -> None:
        assert orchestrator.name == ProviderName.ANTHROPIC
