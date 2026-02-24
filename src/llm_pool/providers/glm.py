from __future__ import annotations

from llm_pool.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig


class GlmAdapter(OpenAICompatAdapter):
    """GLM adapter via OpenAI-compatible API surface."""

    def __init__(self) -> None:
        super().__init__(
            name="glm",
            config=OpenAICompatConfig(
                base_url="https://api.z.ai/api/coding/paas/v4",
                api_key_env="ZAI_API_KEY",
            ),
        )
