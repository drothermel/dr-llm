from __future__ import annotations

from dr_llm.llm.catalog.model_blacklist import (
    AVOID_MORE_EXPENSIVE_BUT_FASTER_MODELS,
    GOOGLE_GEMINI_20_FLASH_UNAVAILABLE,
    GOOGLE_IRRELEVANT_MODELS,
    IRRELEVANT_FOR_RESEARCH,
    OPENAI_LANGUAGE_MODEL_PRICING,
    blacklist_reason,
)


def test_openai_language_model_pricing_contains_requested_models() -> None:
    expected_models = {
        "gpt-5.4-2026-03-05",
        "gpt-5.4",
        "gpt-5.2-2025-12-11",
        "gpt-5.2",
        "gpt-5.1-2025-11-13",
        "gpt-5.1",
        "gpt-5-chat-latest",
        "gpt-5-2025-08-07",
        "gpt-5",
        "gpt-4.1-2025-04-14",
        "gpt-4.1",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o",
        "gpt-4-0613",
        "gpt-4",
        "o3-2025-04-16",
        "o3",
        "o1-2024-12-17",
        "o1",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex",
        "gpt-5-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-mini",
        "gpt-5-mini-2025-08-07",
        "gpt-5-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-mini",
        "o4-mini-2025-04-16",
        "o4-mini",
        "o3-mini",
        "o3-mini-2025-01-31",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.4-nano",
        "gpt-5-nano-2025-08-07",
        "gpt-5-nano",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4.1-nano",
    }

    assert set(OPENAI_LANGUAGE_MODEL_PRICING) == expected_models


def test_openai_language_model_pricing_values() -> None:
    assert OPENAI_LANGUAGE_MODEL_PRICING["gpt-5.4"].input_cost_per_1m == 2.5
    assert OPENAI_LANGUAGE_MODEL_PRICING["gpt-5.4"].output_cost_per_1m == 15.0
    assert OPENAI_LANGUAGE_MODEL_PRICING["gpt-4"].input_cost_per_1m == 30.0
    assert OPENAI_LANGUAGE_MODEL_PRICING["gpt-4"].output_cost_per_1m == 60.0
    assert OPENAI_LANGUAGE_MODEL_PRICING["gpt-5.1-codex-mini"].input_cost_per_1m == 0.25
    assert OPENAI_LANGUAGE_MODEL_PRICING["gpt-5.1-codex-mini"].output_cost_per_1m == 2.0


def test_google_irrelevant_models_are_blacklisted() -> None:
    for model in GOOGLE_IRRELEVANT_MODELS:
        assert blacklist_reason(provider="google", model=model) == IRRELEVANT_FOR_RESEARCH


def test_google_gemini_20_flash_models_are_blacklisted_as_unavailable() -> None:
    for model in (
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
    ):
        assert (
            blacklist_reason(provider="google", model=model)
            == GOOGLE_GEMINI_20_FLASH_UNAVAILABLE
        )


def test_glm_5_turbo_is_blacklisted() -> None:
    assert (
        blacklist_reason(provider="glm", model="glm-5-turbo")
        == AVOID_MORE_EXPENSIVE_BUT_FASTER_MODELS
    )
