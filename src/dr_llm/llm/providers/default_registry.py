from __future__ import annotations

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.anthropic.orchestrator import (
    AnthropicOrchestrator,
)
from dr_llm.llm.providers.impls.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.impls.claude_code.orchestrator import (
    ClaudeCodeOrchestrator,
)
from dr_llm.llm.providers.impls.claude_code.provider import (
    ClaudeCodeProvider,
)
from dr_llm.llm.providers.impls.codex.orchestrator import (
    CodexOrchestrator,
)
from dr_llm.llm.providers.impls.codex.provider import CodexProvider
from dr_llm.llm.providers.impls.glm.orchestrator import GlmOrchestrator
from dr_llm.llm.providers.impls.google.orchestrator import GoogleOrchestrator
from dr_llm.llm.providers.impls.google.provider import GoogleProvider
from dr_llm.llm.providers.impls.kimi_code.orchestrator import (
    KimiCodeOrchestrator,
)
from dr_llm.llm.providers.impls.kimi_code.provider import KimiCodeProvider
from dr_llm.llm.providers.impls.minimax.orchestrator import (
    MiniMaxOrchestrator,
)
from dr_llm.llm.providers.impls.minimax.provider import MiniMaxProvider
from dr_llm.llm.providers.impls.openai.orchestrator import OpenAIOrchestrator
from dr_llm.llm.providers.impls.openrouter.orchestrator import (
    OpenRouterOrchestrator,
)
from dr_llm.llm.providers.transports.api_config import APIProviderConfig
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from dr_llm.llm.providers.transports.openai_compat.provider import (
    OpenAICompatProvider,
)

_OPENAI_COMPAT_PROVIDERS: tuple[tuple[ProviderName, str, str], ...] = (
    (ProviderName.OPENAI, "https://api.openai.com/v1", "OPENAI_API_KEY"),
    (
        ProviderName.OPENROUTER,
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    ),
    (ProviderName.GLM, "https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
)

_OPENAI_COMPAT_ORCHESTRATORS = {
    ProviderName.OPENAI: OpenAIOrchestrator,
    ProviderName.OPENROUTER: OpenRouterOrchestrator,
    ProviderName.GLM: GlmOrchestrator,
}


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()

    for name, base_url, api_key_env in _OPENAI_COMPAT_PROVIDERS:
        provider = OpenAICompatProvider(
            config=OpenAICompatConfig(
                name=name,
                base_url=base_url,
                api_key_env=api_key_env,
            ),
        )
        registry.register(_OPENAI_COMPAT_ORCHESTRATORS[name](provider))

    minimax = MiniMaxProvider()
    registry.register(MiniMaxOrchestrator(minimax))

    anthropic = AnthropicProvider()
    registry.register(AnthropicOrchestrator(anthropic))

    google = GoogleProvider(
        config=APIProviderConfig(
            name=ProviderName.GOOGLE,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GOOGLE_API_KEY",
        )
    )
    registry.register(GoogleOrchestrator(google))

    codex = CodexProvider()
    registry.register(CodexOrchestrator(codex))

    claude_code = ClaudeCodeProvider()
    registry.register(ClaudeCodeOrchestrator(claude_code))

    kimi_code = KimiCodeProvider()
    registry.register(KimiCodeOrchestrator(kimi_code))

    return registry
