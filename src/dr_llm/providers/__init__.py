from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.google.adapter import GoogleAdapter
from dr_llm.providers.headless.claude import ClaudeHeadlessAdapter
from dr_llm.providers.headless.codex import CodexHeadlessAdapter
from dr_llm.providers.kimi_code import KimiCodeAdapter
from dr_llm.providers.minimax import MiniMaxAdapter
from dr_llm.providers.openai_compat.adapter import OpenAICompatAdapter
from dr_llm.providers.openai_compat.config import OpenAICompatConfig
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
    registry.register(MiniMaxAdapter())
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
    registry.register(
        GoogleAdapter(
            config=APIProviderConfig(
                name="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                api_key_env="GOOGLE_API_KEY",
            )
        )
    )
    registry.register(CodexHeadlessAdapter())
    registry.register(ClaudeHeadlessAdapter())
    registry.register(KimiCodeAdapter())
    return registry
