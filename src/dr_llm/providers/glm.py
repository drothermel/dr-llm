from __future__ import annotations

from dr_llm.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig


class GlmAdapter(OpenAICompatAdapter):
    """GLM adapter via OpenAI-compatible API surface."""

    def __init__(self) -> None:
        super().__init__(
            config=OpenAICompatConfig(
                name="glm",
                base_url="https://api.z.ai/api/coding/paas/v4",
                api_key_env="ZAI_API_KEY",
            ),
        )
