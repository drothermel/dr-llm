from __future__ import annotations

from collections.abc import Callable

from dr_llm.llm.names import ProviderName, ReasoningMode
from dr_llm.llm.providers.anthropic.capabilities import (
    reasoning_capabilities_for_anthropic,
)
from dr_llm.llm.providers.concepts.capabilities import (
    GoogleThinkingLevel,
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.google.capabilities import (
    reasoning_capabilities_for_google,
)
from dr_llm.llm.providers.headless.claude_capabilities import (
    reasoning_capabilities_for_claude_code,
)
from dr_llm.llm.providers.headless.codex_thinking import (
    reasoning_capabilities_for_codex,
)
from dr_llm.llm.providers.kimi_code_capabilities import (
    reasoning_capabilities_for_kimi_code,
)
from dr_llm.llm.providers.minimax_capabilities import (
    reasoning_capabilities_for_minimax,
)
from dr_llm.llm.providers.openai_compat.glm_capabilities import (
    reasoning_capabilities_for_glm,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    reasoning_capabilities_for_openai,
)
from dr_llm.llm.providers.openrouter.policy import (
    reasoning_capabilities_for_openrouter,
)

__all__ = [
    "GoogleThinkingLevel",
    "ReasoningCapabilities",
    "ReasoningCapabilityRule",
    "ReasoningMode",
    "reasoning_capabilities_for_model",
    "resolve_capability_rules",
]


CapabilityResolver = Callable[[str], ReasoningCapabilities | None]

_CAPABILITY_RESOLVERS: dict[str, CapabilityResolver] = {
    ProviderName.ANTHROPIC: reasoning_capabilities_for_anthropic,
    ProviderName.CLAUDE_CODE: reasoning_capabilities_for_claude_code,
    ProviderName.CODEX: reasoning_capabilities_for_codex,
    ProviderName.GLM: reasoning_capabilities_for_glm,
    ProviderName.GOOGLE: reasoning_capabilities_for_google,
    ProviderName.KIMI_CODE: reasoning_capabilities_for_kimi_code,
    ProviderName.MINIMAX: reasoning_capabilities_for_minimax,
    ProviderName.OPENAI: reasoning_capabilities_for_openai,
    ProviderName.OPENROUTER: reasoning_capabilities_for_openrouter,
}


def reasoning_capabilities_for_model(
    *,
    provider: str,
    model: str,
) -> ReasoningCapabilities | None:
    resolver = _CAPABILITY_RESOLVERS.get(provider)
    if resolver is None:
        return None
    return resolver(model)
