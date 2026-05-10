from __future__ import annotations

from dr_llm.llm.catalog.model_blacklist import (
    OpenAIModelPrice,
    blacklist_reason,
    openai_language_model_pricing,
)


def test_openai_language_model_pricing_values() -> None:
    pricing = openai_language_model_pricing()
    assert pricing["gpt-5.4"].input_cost_per_1m == 2.5
    assert pricing["gpt-5.4"].output_cost_per_1m == 15.0
    assert pricing["gpt-4"].input_cost_per_1m == 30.0
    assert pricing["gpt-4"].output_cost_per_1m == 60.0
    assert pricing["gpt-5.1-codex-mini"].input_cost_per_1m == 0.25
    assert pricing["gpt-5.1-codex-mini"].output_cost_per_1m == 2.0


def test_openai_language_model_pricing_returns_a_fresh_copy() -> None:
    pricing = openai_language_model_pricing()
    removed = pricing.pop("gpt-5.4")

    latest_pricing = openai_language_model_pricing()

    assert isinstance(removed, OpenAIModelPrice)
    assert "gpt-5.4" in latest_pricing
    assert latest_pricing is not pricing


def test_google_irrelevant_models_are_blacklisted() -> None:
    expected_reason = "Irrelevant to LLM research."
    for model in (
        "gemini-2.5-flash-image",
        "gemini-embedding-001",
        "veo-3.1-generate-preview",
    ):
        assert (
            blacklist_reason(provider="google", model=model) == expected_reason
        )


def test_google_gemini_20_flash_models_are_blacklisted_as_unavailable() -> (
    None
):
    expected_reason = (
        "No longer available to new users via the Google API as of 2026-04-03. "
        "Use a newer Gemini Flash model instead."
    )
    for model in (
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
    ):
        assert (
            blacklist_reason(provider="google", model=model) == expected_reason
        )


def test_glm_5_turbo_is_blacklisted() -> None:
    assert (
        blacklist_reason(provider="glm", model="glm-5-turbo")
        == "Avoid calling more expensive but faster models."
    )


def test_anthropic_claude_3_haiku_is_blacklisted() -> None:
    expected_reason = (
        "Deprecated by Anthropic on 2026-02-19 and scheduled to retire on 2026-04-20. "
        "Recommended replacement: claude-haiku-4-5-20251001."
    )
    assert (
        blacklist_reason(provider="anthropic", model="claude-3-haiku-20240307")
        == expected_reason
    )


def test_blacklist_yaml_loads_and_validates() -> None:
    from dr_llm.llm.catalog.model_blacklist import _blacklist

    blacklist = _blacklist()
    assert len(blacklist) > 0
    for (provider, model), reason in blacklist.items():
        assert isinstance(provider, str)
        assert provider
        assert isinstance(model, str)
        assert model
        assert isinstance(reason, str)
        assert reason


def test_openai_pricing_yaml_loads_and_validates() -> None:
    pricing = openai_language_model_pricing()
    assert len(pricing) > 0
    for model, price in pricing.items():
        assert isinstance(model, str)
        assert model
        assert price.input_cost_per_1m >= 0
        assert price.output_cost_per_1m >= 0
