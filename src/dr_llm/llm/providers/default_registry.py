from __future__ import annotations

from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.anthropic.orchestrator import (
    AnthropicOrchestrator,
)
from dr_llm.llm.providers.impls.claude_code.orchestrator import (
    ClaudeCodeOrchestrator,
)
from dr_llm.llm.providers.impls.codex.orchestrator import (
    CodexOrchestrator,
)
from dr_llm.llm.providers.impls.glm.orchestrator import GlmOrchestrator
from dr_llm.llm.providers.impls.google.orchestrator import GoogleOrchestrator
from dr_llm.llm.providers.impls.kimi_code.orchestrator import (
    KimiCodeOrchestrator,
)
from dr_llm.llm.providers.impls.minimax.orchestrator import (
    MiniMaxOrchestrator,
)
from dr_llm.llm.providers.impls.openai.orchestrator import OpenAIOrchestrator
from dr_llm.llm.providers.impls.openrouter.orchestrator import (
    OpenRouterOrchestrator,
)


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()

    registry.register(OpenAIOrchestrator())
    registry.register(OpenRouterOrchestrator())
    registry.register(GlmOrchestrator())
    registry.register(MiniMaxOrchestrator())
    registry.register(AnthropicOrchestrator())
    registry.register(GoogleOrchestrator())
    registry.register(CodexOrchestrator())
    registry.register(ClaudeCodeOrchestrator())
    registry.register(KimiCodeOrchestrator())

    return registry
