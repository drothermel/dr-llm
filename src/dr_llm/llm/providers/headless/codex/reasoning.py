from __future__ import annotations

from pydantic import Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.headless.codex.capabilities import (
    codex_supports_configurable_thinking,
    codex_supports_minimal_thinking,
    codex_supports_off_thinking,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    CodexReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    is_reasoning_unsupported,
    unsupported_reasoning_kind_message,
    validate_budget_range,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.reasoning_capabilities import (
    reasoning_capabilities_for_model,
)


def validate_reasoning_for_codex(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    def _validate_native(spec: CodexReasoning) -> None:
        if not codex_supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.CODEX} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.CODEX,
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=codex_supports_off_thinking(model),
            supports_minimal=codex_supports_minimal_thinking(model),
            supports_xhigh=True,
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        capabilities = reasoning_capabilities_for_model(
            provider=ProviderName.CODEX, model=model
        )
        if is_reasoning_unsupported(capabilities):
            raise ValueError(
                f"Reasoning is not supported for provider='{ProviderName.CODEX}' model={model!r}"
            )
        assert capabilities is not None
        validate_budget_range(
            provider=ProviderName.CODEX,
            model=model,
            label="reasoning budget",
            tokens=budget.tokens,
            capabilities=capabilities,
        )

    dispatch_reasoning_validation(
        provider=ProviderName.CODEX,
        model=model,
        reasoning=reasoning,
        native_spec_type=CodexReasoning,
        requires_reasoning=codex_supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )


class CodexHeadlessReasoningConfig(BaseProviderReasoningConfig):
    cli_args: list[str] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> CodexHeadlessReasoningConfig:
        if config is None:
            return cls()
        match config:
            case CodexReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case CodexReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(cli_args=["-c", 'model_reasoning_effort="none"'])
            case CodexReasoning(
                thinking_level=ThinkingLevel.MINIMAL
                | ThinkingLevel.LOW
                | ThinkingLevel.MEDIUM
                | ThinkingLevel.HIGH
                | ThinkingLevel.XHIGH
            ):
                thinking_level = config.thinking_level
                return cls(
                    cli_args=[
                        "-c",
                        f'model_reasoning_effort="{thinking_level}"',
                    ]
                )
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("codex headless", config)
        )
