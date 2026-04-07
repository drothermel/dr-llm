from __future__ import annotations

import httpx

from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.anthropic.config import AnthropicConfig

MINIMAX_PROVIDER_NAME = "minimax"
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"


class MiniMaxProvider(AnthropicProvider):
    """Anthropic Messages API compatibility layer for MiniMax.

    MiniMax exposes an Anthropic-shaped HTTP API, but only a subset of Anthropic
    features apply: the service targets the MiniMax **M2.x** text generation
    models (older/other families are not available on this endpoint). Image and
    document modalities are **not** supported because the gateway does not
    implement Anthropic's multimodal content blocks. Extended thinking and
    adaptive budgets are unavailable: the API does not expose Anthropic-style
    thinking/reasoning controls, so callers should treat reasoning as ``na`` and
    rely on provider defaults. Token limits differ from upstream Anthropic
    (``max_tokens`` is optional or validated more loosely) because request
    validation follows MiniMax's rules, not Anthropic's.
    """

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        # Default config: Anthropic-compatible URL and MiniMax API key env (see class docstring).
        super().__init__(
            config=config
            or AnthropicConfig(
                name=MINIMAX_PROVIDER_NAME,
                base_url=MINIMAX_BASE_URL,
                api_key_env=MINIMAX_API_KEY_ENV,
            ),
            client=client,
        )
