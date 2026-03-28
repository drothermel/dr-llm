from dr_llm.providers.anthropic import AnthropicAdapter
from dr_llm.providers.avail import (
    ProviderAvailabilityStatus,
    available_provider_names,
    supported_provider_names,
    supported_provider_statuses,
)
from dr_llm.providers.glm import GlmAdapter
from dr_llm.providers.google import GoogleAdapter
from dr_llm.providers.headless import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from dr_llm.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from dr_llm.providers.registry import ProviderRegistry


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    openai_config = OpenAICompatConfig(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    )
    registry.register(
        OpenAICompatAdapter(
            name="openai",
            config=openai_config,
        )
    )
    registry.register(
        OpenAICompatAdapter(
            name="openrouter",
            config=OpenAICompatConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
            ),
        )
    )
    registry.register(
        OpenAICompatAdapter(
            name="minimax",
            config=OpenAICompatConfig(
                base_url="https://api.minimax.io/v1",
                api_key_env="MINIMAX_API_KEY",
            ),
        )
    )
    registry.register(AnthropicAdapter())
    registry.register(GoogleAdapter())
    registry.register(GlmAdapter())
    registry.register(CodexHeadlessAdapter())
    registry.register(ClaudeHeadlessAdapter())
    registry.register(ClaudeHeadlessMiniMaxAdapter())
    registry.register(ClaudeHeadlessKimiAdapter())
    return registry


__all__ = [
    "AnthropicAdapter",
    "ProviderAvailabilityStatus",
    "ClaudeHeadlessAdapter",
    "ClaudeHeadlessKimiAdapter",
    "ClaudeHeadlessMiniMaxAdapter",
    "CodexHeadlessAdapter",
    "GlmAdapter",
    "GoogleAdapter",
    "OpenAICompatAdapter",
    "OpenAICompatConfig",
    "ProviderRegistry",
    "available_provider_names",
    "build_default_registry",
    "supported_provider_names",
    "supported_provider_statuses",
]
