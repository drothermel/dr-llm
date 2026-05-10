from __future__ import annotations

from collections.abc import Callable

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.anthropic.reasoning import (
    validate_reasoning_for_anthropic,
    validate_reasoning_for_kimi_code,
    validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.google.reasoning import validate_reasoning_for_google
from dr_llm.llm.providers.headless.claude.reasoning import (
    validate_reasoning_for_claude_code,
)
from dr_llm.llm.providers.headless.codex.reasoning import (
    validate_reasoning_for_codex,
)
from dr_llm.llm.providers.openai_compat.reasoning import (
    validate_reasoning_for_glm,
    validate_reasoning_for_openai,
    validate_reasoning_for_openrouter,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec

ReasoningValidator = Callable[..., None]

_REASONING_VALIDATORS: dict[str, ReasoningValidator] = {
    ProviderName.ANTHROPIC: validate_reasoning_for_anthropic,
    ProviderName.CLAUDE_CODE: validate_reasoning_for_claude_code,
    ProviderName.CODEX: validate_reasoning_for_codex,
    ProviderName.GLM: validate_reasoning_for_glm,
    ProviderName.GOOGLE: validate_reasoning_for_google,
    ProviderName.KIMI_CODE: validate_reasoning_for_kimi_code,
    ProviderName.MINIMAX: validate_reasoning_for_minimax,
    ProviderName.OPENAI: validate_reasoning_for_openai,
    ProviderName.OPENROUTER: validate_reasoning_for_openrouter,
}


def validate_reasoning(
    *,
    provider: str,
    model: str,
    reasoning: ReasoningSpec | None,
) -> None:
    validator = _REASONING_VALIDATORS.get(provider)
    if validator is None:
        if reasoning is not None:
            raise ValueError(
                f"reasoning is not supported for provider={provider!r}"
            )
        return
    validator(model=model, reasoning=reasoning)
