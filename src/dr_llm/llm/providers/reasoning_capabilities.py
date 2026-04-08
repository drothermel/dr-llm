from __future__ import annotations

from collections.abc import Callable

from dr_llm.llm.providers.anthropic.capabilities import (
    reasoning_capabilities_for_anthropic,
)
from dr_llm.llm.providers.google.capabilities import reasoning_capabilities_for_google
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
from dr_llm.llm.providers.reasoning_capability_types import (
    GoogleThinkingLevel,
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    ReasoningMode,
    resolve_capability_rules,
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
    "anthropic": reasoning_capabilities_for_anthropic,
    "claude-code": reasoning_capabilities_for_claude_code,
    "codex": reasoning_capabilities_for_codex,
    "glm": reasoning_capabilities_for_glm,
    "google": reasoning_capabilities_for_google,
    "kimi-code": reasoning_capabilities_for_kimi_code,
    "minimax": reasoning_capabilities_for_minimax,
    "openai": reasoning_capabilities_for_openai,
    "openrouter": reasoning_capabilities_for_openrouter,
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
