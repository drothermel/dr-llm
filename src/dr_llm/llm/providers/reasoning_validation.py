from __future__ import annotations

from collections.abc import Callable

from dr_llm.llm.providers.anthropic.reasoning import (
    validate_reasoning_for_anthropic,
    validate_reasoning_for_kimi_code,
    validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.google.reasoning import validate_reasoning_for_google
from dr_llm.llm.providers.headless.reasoning import (
    validate_reasoning_for_claude_code,
    validate_reasoning_for_codex,
)
from dr_llm.llm.providers.openai_compat.reasoning import (
    validate_reasoning_for_glm,
    validate_reasoning_for_openai,
    validate_reasoning_for_openrouter,
)
from dr_llm.llm.providers.reasoning import ReasoningSpec

ReasoningValidator = Callable[..., None]

_REASONING_VALIDATORS: dict[str, ReasoningValidator] = {
    "anthropic": validate_reasoning_for_anthropic,
    "claude-code": validate_reasoning_for_claude_code,
    "codex": validate_reasoning_for_codex,
    "glm": validate_reasoning_for_glm,
    "google": validate_reasoning_for_google,
    "kimi-code": validate_reasoning_for_kimi_code,
    "minimax": validate_reasoning_for_minimax,
    "openai": validate_reasoning_for_openai,
    "openrouter": validate_reasoning_for_openrouter,
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
            raise ValueError(f"reasoning is not supported for provider={provider!r}")
        return
    validator(model=model, reasoning=reasoning)
