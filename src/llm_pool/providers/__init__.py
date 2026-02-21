from llm_pool.providers.anthropic import AnthropicAdapter
from llm_pool.providers.glm import GlmAdapter
from llm_pool.providers.google import GoogleAdapter
from llm_pool.providers.headless import ClaudeHeadlessAdapter, CodexHeadlessAdapter
from llm_pool.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from llm_pool.providers.registry import ProviderRegistry


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(
        OpenAICompatAdapter(
            name="openai",
            config=OpenAICompatConfig(
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
            ),
        )
    )
    registry.register(
        OpenAICompatAdapter(
            name="openai-compatible",
            config=OpenAICompatConfig(
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
            ),
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
    registry.register(AnthropicAdapter())
    registry.register(GoogleAdapter())
    registry.register(GlmAdapter())
    registry.register(CodexHeadlessAdapter(), aliases=["codex-cli"])
    registry.register(ClaudeHeadlessAdapter(), aliases=["claude", "claude-code"])
    return registry


__all__ = [
    "AnthropicAdapter",
    "ClaudeHeadlessAdapter",
    "CodexHeadlessAdapter",
    "GlmAdapter",
    "GoogleAdapter",
    "OpenAICompatAdapter",
    "OpenAICompatConfig",
    "ProviderRegistry",
    "build_default_registry",
]
