from __future__ import annotations

from dr_llm.llm import (
    Message,
    ProviderCategories,
    ProviderName,
    parse_llm_config,
    parse_llm_request,
)


def test_provider_name_members_behave_like_strings() -> None:
    assert ProviderName.OPENAI == str(ProviderName.OPENAI)
    assert isinstance(ProviderName.OPENAI, str)


def test_provider_categories_group_provider_names() -> None:
    provider_cats = ProviderCategories()

    assert provider_cats.openai == (ProviderName.OPENAI,)
    assert provider_cats.sampling_api == (
        ProviderName.OPENROUTER,
        ProviderName.GLM,
        ProviderName.GOOGLE,
        ProviderName.ANTHROPIC,
        ProviderName.MINIMAX,
    )
    assert provider_cats.kimi_code == (ProviderName.KIMI_CODE,)
    assert provider_cats.api_backed == (
        ProviderName.OPENAI,
        ProviderName.OPENROUTER,
        ProviderName.GLM,
        ProviderName.GOOGLE,
        ProviderName.ANTHROPIC,
        ProviderName.MINIMAX,
        ProviderName.KIMI_CODE,
    )
    assert provider_cats.headless == (
        ProviderName.CODEX,
        ProviderName.CLAUDE_CODE,
    )


def test_provider_categories_can_be_overridden() -> None:
    provider_cats = ProviderCategories(
        api_backed=(ProviderName.OPENAI,),
        headless=(ProviderName.CODEX,),
    )

    assert provider_cats.api_backed == (ProviderName.OPENAI,)
    assert provider_cats.headless == (ProviderName.CODEX,)
    assert provider_cats.sampling_api == ProviderCategories().sampling_api


def test_provider_name_discriminators_accept_raw_strings() -> None:
    config = parse_llm_config(
        {"provider": str(ProviderName.OPENAI), "model": "gpt-4.1-mini"}
    )
    request = parse_llm_request(
        {
            "provider": str(ProviderName.OPENAI),
            "model": "gpt-4.1-mini",
            "messages": [Message(role="user", content="hi")],
        }
    )

    assert config.provider is ProviderName.OPENAI
    assert request.provider is ProviderName.OPENAI
    assert config.model_dump(mode="json")["provider"] == str(
        ProviderName.OPENAI
    )
    assert request.model_dump(mode="json")["provider"] == str(
        ProviderName.OPENAI
    )
