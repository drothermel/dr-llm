from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderControlMapping,
    GoogleReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    google_literal_to_thinking_level,
    is_control_unsupported,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
    validate_allowed_thinking_levels,
    validate_budget_range,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class GoogleThinkingLevel(StrEnum):
    MINIMAL = ThinkingLevel.MINIMAL
    LOW = ThinkingLevel.LOW
    MEDIUM = ThinkingLevel.MEDIUM
    HIGH = ThinkingLevel.HIGH


class GoogleModelFamily(StrEnum):
    GEMINI_25_FLASH_LITE_PREVIEW = "gemini-2.5-flash-lite-preview"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_25_FLASH_PREVIEW = "gemini-2.5-flash-preview"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_3 = "gemini-3"
    GEMMA_4 = "gemma-4"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


GOOGLE_25_FLASH_LITE_FAMILIES = (
    GoogleModelFamily.GEMINI_25_FLASH_LITE_PREVIEW,
    GoogleModelFamily.GEMINI_25_FLASH_LITE,
)
GOOGLE_25_FLASH_FAMILIES = (
    GoogleModelFamily.GEMINI_25_FLASH_PREVIEW,
    GoogleModelFamily.GEMINI_25_FLASH,
)
GOOGLE_25_PRO_FAMILIES = (GoogleModelFamily.GEMINI_25_PRO,)
GOOGLE_3_FAMILIES = (GoogleModelFamily.GEMINI_3,)
GEMMA_4_FAMILIES = (GoogleModelFamily.GEMMA_4,)


class GoogleMinBudget(IntEnum):
    GEMINI_25_FLASH = 1
    GEMINI_25_FLASH_LITE = 512
    GEMINI_25_PRO = 128


class GoogleMaxBudget(IntEnum):
    GEMINI_25_FLASH = 24576
    GEMINI_25_FLASH_LITE = 24576
    GEMINI_25_PRO = 32768


def google_control_mode(model: str) -> ControlMode:
    if any(
        family.in_family(model)
        for family in (
            *GOOGLE_25_FLASH_LITE_FAMILIES,
            *GOOGLE_25_FLASH_FAMILIES,
            *GOOGLE_25_PRO_FAMILIES,
        )
    ):
        return ControlMode.GOOGLE_BUDGET
    if any(
        family.in_family(model)
        for family in (*GOOGLE_3_FAMILIES, *GEMMA_4_FAMILIES)
    ):
        return ControlMode.GOOGLE_LEVEL
    return ControlMode.UNSUPPORTED


# Google Generative Language API `thinkingBudget` sentinel values.
_GOOGLE_THINKING_BUDGET_OFF = 0
_GOOGLE_THINKING_BUDGET_ADAPTIVE = -1


def validate_reasoning_for_google(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    controls = GoogleControls(model=model, mode=CallMode.api)

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        if controls.control_mode == ControlMode.UNSUPPORTED:
            raise ValueError(
                f"Reasoning is not allowed for provider='{ProviderName.GOOGLE}' model={model!r}: reasoning capabilities are unknown"
            )
        if controls.control_mode == ControlMode.GOOGLE_LEVEL:
            raise ValueError(
                f"Top-level reasoning budget is not supported for provider='{ProviderName.GOOGLE}' model={model!r} with control_mode={controls.control_mode!r}"
            )
        validate_budget_range(
            provider=ProviderName.GOOGLE,
            model=model,
            label="reasoning budget",
            tokens=budget.tokens,
            min_budget_tokens=controls.min_budget_tokens,
            max_budget_tokens=controls.max_budget_tokens,
        )

    dispatch_reasoning_validation(
        provider=ProviderName.GOOGLE,
        model=model,
        reasoning=reasoning,
        native_spec_type=GoogleReasoning,
        requires_reasoning=not is_control_unsupported(controls.control_mode),
        validate_native=lambda spec: _validate_google_reasoning_shape(
            model=model,
            thinking_level=spec.thinking_level,
            budget_tokens=spec.budget_tokens,
        ),
        validate_top_budget=_validate_top_budget,
    )


def _validate_google_reasoning_shape(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> None:
    controls = GoogleControls(model=model, mode=CallMode.api)
    if is_control_unsupported(controls.control_mode):
        if thinking_level == ThinkingLevel.NA:
            return
        raise ValueError(
            f"{ProviderName.GOOGLE} thinking is not supported for model={model!r}"
        )
    if thinking_level == ThinkingLevel.NA:
        raise ValueError(
            f"thinking_level='na' is not supported for provider='{ProviderName.GOOGLE}' model={model!r}"
        )
    if controls.control_mode == ControlMode.GOOGLE_BUDGET:
        _validate_google_budget_mode(
            model=model,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            controls=controls,
        )
        return
    if controls.control_mode == ControlMode.GOOGLE_LEVEL:
        _validate_google_level_mode(
            model=model,
            thinking_level=thinking_level,
            controls=controls,
        )
        return
    raise ValueError(
        f"Reasoning is not supported for provider='{ProviderName.GOOGLE}' model={model!r}"
    )


def _validate_google_budget_mode(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
    controls: "GoogleControls",
) -> None:
    if thinking_level == ThinkingLevel.OFF:
        return
    if thinking_level == ThinkingLevel.ADAPTIVE:
        if controls.supports_dynamic:
            return
        raise ValueError(
            f"{ProviderName.GOOGLE} dynamic thinking is not supported for model={model!r}"
        )
    if thinking_level == ThinkingLevel.BUDGET:
        if budget_tokens is None:
            raise ValueError(
                "google budget thinking requires budget_tokens when "
                "thinking_level is 'budget'"
            )
        validate_budget_range(
            provider=ProviderName.GOOGLE,
            model=model,
            label=f"{ProviderName.GOOGLE} thinking_budget",
            tokens=budget_tokens,
            min_budget_tokens=controls.min_budget_tokens,
            max_budget_tokens=controls.max_budget_tokens,
        )
        return
    raise ValueError(
        f"{ProviderName.GOOGLE} model {model!r} does not support thinking_level={thinking_level!r}; use off, adaptive, or budget"
    )


def _validate_google_level_mode(
    *,
    model: str,
    thinking_level: ThinkingLevel,
    controls: "GoogleControls",
) -> None:
    allowed_levels = {
        google_literal_to_thinking_level(level)
        for level in controls.google_thinking_levels
    }
    validate_allowed_thinking_levels(
        provider=ProviderName.GOOGLE,
        model=model,
        thinking_level=thinking_level,
        allowed_levels=allowed_levels,
        allow_na=False,
    )


class GoogleControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.GOOGLE
    model: str
    mode: CallMode

    @property
    def control_mode(self) -> ControlMode:
        return google_control_mode(self.model)

    @property
    def min_budget_tokens(self) -> int | None:
        if any(
            family.in_family(self.model)
            for family in GOOGLE_25_FLASH_LITE_FAMILIES
        ):
            return GoogleMinBudget.GEMINI_25_FLASH_LITE
        if any(
            family.in_family(self.model) for family in GOOGLE_25_FLASH_FAMILIES
        ):
            return GoogleMinBudget.GEMINI_25_FLASH
        if any(
            family.in_family(self.model) for family in GOOGLE_25_PRO_FAMILIES
        ):
            return GoogleMinBudget.GEMINI_25_PRO
        return None

    @property
    def max_budget_tokens(self) -> int | None:
        if any(
            family.in_family(self.model)
            for family in GOOGLE_25_FLASH_LITE_FAMILIES
        ):
            return GoogleMaxBudget.GEMINI_25_FLASH_LITE
        if any(
            family.in_family(self.model) for family in GOOGLE_25_FLASH_FAMILIES
        ):
            return GoogleMaxBudget.GEMINI_25_FLASH
        if any(
            family.in_family(self.model) for family in GOOGLE_25_PRO_FAMILIES
        ):
            return GoogleMaxBudget.GEMINI_25_PRO
        return None

    @property
    def supports_dynamic(self) -> bool:
        return self.control_mode == ControlMode.GOOGLE_BUDGET

    @property
    def google_thinking_levels(self) -> tuple[GoogleThinkingLevel, ...]:
        if any(family.in_family(self.model) for family in GOOGLE_3_FAMILIES):
            return (
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.LOW,
                GoogleThinkingLevel.MEDIUM,
                GoogleThinkingLevel.HIGH,
            )
        if any(family.in_family(self.model) for family in GEMMA_4_FAMILIES):
            return (
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.HIGH,
            )
        return ()

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if self.control_mode == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if self.control_mode == ControlMode.GOOGLE_BUDGET:
            return (
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.OFF,
                ThinkingLevel.BUDGET,
            )
        if self.control_mode == ControlMode.GOOGLE_LEVEL:
            return tuple(
                google_literal_to_thinking_level(level)
                for level in self.google_thinking_levels
            )
        raise ValueError(
            f"unexpected control mode for provider={self.provider!r} "
            f"model={self.model!r}: {self.control_mode!r}"
        )

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        levels = self.supported_thinking_levels
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.ADAPTIVE,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return ()

    @property
    def default_effort(self) -> EffortSpec:
        return EffortSpec.NA

    @property
    def default_reasoning(self) -> ReasoningSpec | None:
        return self.reasoning_for_thinking_level(
            thinking_level=self.default_thinking_level,
            budget_tokens=self.min_budget_tokens,
        )

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        return {
            "control_mode": self.control_mode,
            "min_budget_tokens": self.min_budget_tokens,
            "max_budget_tokens": self.max_budget_tokens,
            "supports_dynamic": self.supports_dynamic,
            "google_thinking_levels": self.google_thinking_levels,
            "supported_thinking_levels": self.supported_thinking_levels,
            "default_thinking_level": self.default_thinking_level,
            "supported_effort_levels": self.supported_effort_levels,
            "default_effort": self.default_effort,
        }

    def request_defaults(self) -> ProviderRequestDefaults:
        return ProviderRequestDefaults(
            provider=self.provider,
            model=self.model,
            mode=self.mode,
            effort=self.default_effort,
            reasoning=self.default_reasoning,
            sampling_supported=True,
            sampling=SamplingControls(temperature=1.0, top_p=0.95),
        )

    def resolve_reasoning(
        self,
        *,
        reasoning: ReasoningSpec | None,
        thinking_level: ThinkingLevel | None,
        budget_tokens: int | None,
    ) -> ReasoningSpec | None:
        if reasoning is not None:
            return reasoning
        if thinking_level is not None:
            return self.reasoning_for_thinking_level(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
            )
        return self.default_reasoning

    def resolve_effort(self, effort: EffortSpec | None) -> EffortSpec:
        if effort is None:
            return self.default_effort
        return effort

    def resolve_sampling(
        self, sampling: SamplingControls | None
    ) -> SamplingControls | None:
        if sampling is not None:
            if sampling.is_empty():
                return None
            return sampling
        return SamplingControls(temperature=1.0, top_p=0.95)

    def reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        if thinking_level == ThinkingLevel.NA:
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=_require_budget_tokens(
                    provider=self.provider,
                    budget_tokens=budget_tokens,
                ),
            )
        return GoogleReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list:
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_google(
            model=request.model, reasoning=request.reasoning
        )
        return []


def _require_budget_tokens(*, provider: str, budget_tokens: int | None) -> int:
    if budget_tokens is None:
        raise ValueError(f"{provider} budget thinking requires budget_tokens")
    return budget_tokens


def _validate_effort(
    *,
    provider: str,
    model: str,
    effort: EffortSpec,
    supported_effort_levels: tuple[EffortSpec, ...],
) -> None:
    if not supported_effort_levels:
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} "
                f"model={model!r}"
            )
        return
    if effort == EffortSpec.NA:
        raise ValueError(
            f"effort is required for provider={provider!r} model={model!r}"
        )
    if effort not in supported_effort_levels:
        allowed = ", ".join(str(level) for level in supported_effort_levels)
        raise ValueError(
            f"effort={effort!r} is not supported for provider={provider!r} "
            f"model={model!r}; allowed levels: {allowed}"
        )


class GoogleControlMapping(BaseProviderControlMapping):
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GoogleControlMapping:
        if config is None:
            return cls()
        match config:
            case ReasoningBudget(tokens=tokens):
                return cls(payload={"thinkingBudget": tokens})
            case GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                include_thoughts=include_thoughts,
            ):
                if thinking_level == ThinkingLevel.NA:
                    return cls()
                payload = _build_thinking_payload(
                    thinking_level=thinking_level,
                    budget_tokens=budget_tokens,
                )
                if include_thoughts is not None:
                    payload["includeThoughts"] = include_thoughts
                return cls(payload=payload)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.GOOGLE, config
                    )
                )


_GOOGLE_LITERAL_LEVELS = {
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
}


def _build_thinking_payload(
    *,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> dict[str, Any]:
    if thinking_level == ThinkingLevel.OFF:
        return {"thinkingBudget": _GOOGLE_THINKING_BUDGET_OFF}
    if thinking_level == ThinkingLevel.ADAPTIVE:
        return {"thinkingBudget": _GOOGLE_THINKING_BUDGET_ADAPTIVE}
    if thinking_level == ThinkingLevel.BUDGET:
        return {
            "thinkingBudget": require_budget_tokens(
                budget_tokens, label=ProviderName.GOOGLE, min_value=0
            )
        }
    if thinking_level in _GOOGLE_LITERAL_LEVELS:
        return {"thinkingLevel": str(thinking_level)}
    raise ProviderSemanticError(
        f"{ProviderName.GOOGLE} reasoning config did not contain a serializable setting"
    )
