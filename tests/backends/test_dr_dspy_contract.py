from __future__ import annotations

from dr_llm.backends import BackendRequest, fingerprint_request
from dr_llm.llm import (
    CallMode,
    GoogleBudgetConfig,
    Message,
    OpenAIGpt5Config,
    OpenRouterEffortConfig,
    OpenRouterEffortLevel,
    OpenRouterToggleConfig,
    ProviderName,
    SamplingControls,
    ThinkingLevel,
)
from dr_llm.llm.config import LlmConfig
from dr_llm.llm.providers.concepts.reasoning import (
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
)
from dr_llm.llm.providers.impls.google.request_controls import (
    GoogleRequestControls,
)
from dr_llm.llm.providers.impls.openai.request_controls import (
    OpenAIRequestControls,
)
from dr_llm.llm.providers.impls.openrouter.request_controls import (
    OpenRouterRequestControls,
)

EXPERIMENT_SAMPLING = SamplingControls(temperature=0.7, top_p=0.95)
PROMPT_MESSAGES = [Message(role="user", content="contract prompt")]


def _backend_request(config: LlmConfig) -> BackendRequest:
    return BackendRequest(
        provider=config.provider,
        model=config.model,
        mode=config.mode,
        messages=PROMPT_MESSAGES,
        max_tokens=config.max_tokens,
        effort=config.effort,
        reasoning=config.reasoning,
        sampling=config.sampling,
    )


def test_dr_dspy_openrouter_toggle_requests_use_native_reasoning() -> None:
    for model in (
        "xiaomi/mimo-v2-flash",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    ):
        config = OpenRouterToggleConfig(
            model=model,
            reasoning_enabled=False,
            sampling=EXPERIMENT_SAMPLING,
        ).to_llm_config()
        request = _backend_request(config)

        assert request.provider == ProviderName.OPENROUTER
        assert request.mode == CallMode.api
        assert request.reasoning == OpenRouterReasoning(enabled=False)
        assert request.sampling == EXPERIMENT_SAMPLING
        assert OpenRouterRequestControls.from_reasoning(
            request.reasoning
        ).extra_body == {"reasoning": {"enabled": False}}


def test_dr_dspy_openrouter_effort_requests_use_native_reasoning() -> None:
    for model in ("openai/gpt-oss-20b", "openai/gpt-5-nano"):
        config = OpenRouterEffortConfig(
            model=model,
            effort=OpenRouterEffortLevel.LOW,
            sampling=EXPERIMENT_SAMPLING,
        ).to_llm_config()
        request = _backend_request(config)

        assert request.provider == ProviderName.OPENROUTER
        assert request.reasoning == OpenRouterReasoning(
            effort=OpenRouterEffortLevel.LOW
        )
        assert request.sampling == EXPERIMENT_SAMPLING
        assert OpenRouterRequestControls.from_reasoning(
            request.reasoning
        ).extra_body == {"reasoning": {"effort": "low"}}


def test_dr_dspy_openai_minimal_request_has_no_sampling_override() -> None:
    config = OpenAIGpt5Config(
        model="gpt-5-nano",
        thinking_level=ThinkingLevel.MINIMAL,
    ).to_llm_config()
    request = _backend_request(config)

    assert request.provider == ProviderName.OPENAI
    assert request.reasoning == OpenAIReasoning(
        thinking_level=ThinkingLevel.MINIMAL
    )
    assert request.sampling is None
    assert (
        OpenAIRequestControls.from_reasoning(
            request.reasoning
        ).reasoning_effort
        == "minimal"
    )


def test_dr_dspy_google_flash_lite_off_request_uses_native_reasoning() -> None:
    config = GoogleBudgetConfig(
        model="gemini-2.5-flash-lite",
        thinking_level=ThinkingLevel.OFF,
        sampling=EXPERIMENT_SAMPLING,
    ).to_llm_config()
    request = _backend_request(config)

    assert request.provider == ProviderName.GOOGLE
    assert request.reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )
    assert request.sampling == EXPERIMENT_SAMPLING
    assert GoogleRequestControls.from_reasoning(request.reasoning).payload == {
        "thinkingBudget": 0
    }


def test_empty_sampling_authoring_input_resolves_to_no_override() -> None:
    config = OpenRouterEffortConfig(
        model="openai/gpt-5-nano",
        effort=OpenRouterEffortLevel.LOW,
        sampling=SamplingControls(temperature=None, top_p=None),
    ).to_llm_config()

    assert config.sampling is None


def test_dr_dspy_fingerprint_contract_for_metadata_and_controls() -> None:
    base = _backend_request(
        OpenRouterEffortConfig(
            model="openai/gpt-5-nano",
            effort=OpenRouterEffortLevel.LOW,
            sampling=EXPERIMENT_SAMPLING,
        ).to_llm_config()
    )
    with_metadata = base.model_copy(
        update={
            "metadata": {"trace_id": "dr-dspy"},
            "extensions": {"audit": {"run": "pilot"}},
        }
    )
    different_reasoning = base.model_copy(
        update={
            "reasoning": OpenRouterReasoning(
                effort=OpenRouterEffortLevel.MEDIUM
            )
        }
    )
    different_sampling = base.model_copy(
        update={"sampling": SamplingControls(temperature=0.2, top_p=0.95)}
    )

    assert fingerprint_request(base) == fingerprint_request(with_metadata)
    assert fingerprint_request(base) != fingerprint_request(
        different_reasoning
    )
    assert fingerprint_request(base) != fingerprint_request(different_sampling)
