from dr_llm.providers.anthropic import AnthropicAdapter
from dr_llm.providers.google import GoogleAdapter
from dr_llm.providers.headless_adapter import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from dr_llm.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.provider_config import ProviderAvailabilityStatus, ProviderConfig
from dr_llm.providers.registry import ProviderRegistry


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(
        OpenAICompatAdapter(
            config=OpenAICompatConfig(
                name="openai",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
            ),
        )
    )
    registry.register(
        OpenAICompatAdapter(
            config=OpenAICompatConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
            ),
        )
    )
    registry.register(
        OpenAICompatAdapter(
            config=OpenAICompatConfig(
                name="minimax",
                base_url="https://api.minimax.io/v1",
                api_key_env="MINIMAX_API_KEY",
            ),
        )
    )
    registry.register(
        OpenAICompatAdapter(
            config=OpenAICompatConfig(
                name="glm",
                base_url="https://api.z.ai/api/coding/paas/v4",
                api_key_env="ZAI_API_KEY",
            ),
        )
    )
    registry.register(AnthropicAdapter())
    registry.register(GoogleAdapter())
    registry.register(CodexHeadlessAdapter())
    registry.register(ClaudeHeadlessAdapter())
    registry.register(ClaudeHeadlessMiniMaxAdapter())
    registry.register(ClaudeHeadlessKimiAdapter())
    return registry


__all__ = [
    "AnthropicAdapter",
    "ProviderAvailabilityStatus",
    "ProviderConfig",
    "ClaudeHeadlessAdapter",
    "ClaudeHeadlessKimiAdapter",
    "ClaudeHeadlessMiniMaxAdapter",
    "CodexHeadlessAdapter",
    "GoogleAdapter",
    "OpenAICompatAdapter",
    "OpenAICompatConfig",
    "ProviderAdapter",
    "ProviderRegistry",
    "build_default_registry",
]
