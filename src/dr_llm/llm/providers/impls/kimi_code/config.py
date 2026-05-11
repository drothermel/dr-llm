from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig
from dr_llm.llm.names import EffortSpec, ProviderName, ThinkingLevel
from dr_llm.llm.providers.core.authoring import (
    build_provider_config,
    validate_budget_range,
)
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.kimi_code.capabilities import (
    reasoning_capabilities_for_kimi_code,
    supported_effort_levels_for_kimi_code,
)
from dr_llm.llm.providers.impls.kimi_code.families import (
    KimiCodeModelFamily,
)

type _KimiCodeThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.ADAPTIVE,
    ThinkingLevel.BUDGET,
]
type _KimiCodeEffort = Literal[
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
]


class KimiCodeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.KIMI_CODE] = ProviderName.KIMI_CODE
    model: str
    max_tokens: int | None = None
    effort: _KimiCodeEffort | None = None
    thinking_level: _KimiCodeThinkingLevel | None = None
    budget_tokens: int | None = None

    @model_validator(mode="after")
    def _validate_controls(self) -> KimiCodeConfig:
        if self.model != KimiCodeModelFamily.KIMI_FOR_CODING:
            raise ValueError(
                f"KimiCodeConfig only supports provider={self.provider!r} "
                f"model={KimiCodeModelFamily.KIMI_FOR_CODING!r}; "
                f"got model={self.model!r}"
            )
        if self.effort is not None:
            allowed = supported_effort_levels_for_kimi_code(self.model)
            if self.effort not in allowed:
                allowed_values = ", ".join(str(level) for level in allowed)
                raise ValueError(
                    f"KimiCodeConfig effort={self.effort!r} is not "
                    f"supported for provider={self.provider!r} "
                    f"model={self.model!r}; allowed levels: {allowed_values}"
                )
        if (
            self.thinking_level == ThinkingLevel.BUDGET
            and self.budget_tokens is None
        ):
            raise ValueError("KimiCodeConfig requires budget_tokens")
        if (
            self.thinking_level != ThinkingLevel.BUDGET
            and self.budget_tokens is not None
        ):
            raise ValueError("budget_tokens requires thinking_level='budget'")
        if self.budget_tokens is not None:
            capabilities = reasoning_capabilities_for_kimi_code(self.model)
            if (
                capabilities is None
                or capabilities.min_budget_tokens is None
                or capabilities.max_budget_tokens is None
            ):
                raise ValueError(
                    f"KimiCodeConfig budget thinking is not supported for "
                    f"model={self.model!r}"
                )
            validate_budget_range(
                provider=self.provider,
                model=self.model,
                budget_tokens=self.budget_tokens,
                min_tokens=capabilities.min_budget_tokens,
                max_tokens=capabilities.max_budget_tokens,
            )
        return self

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            thinking_level=self.thinking_level,
            budget_tokens=self.budget_tokens,
            registry=registry,
        )
