from __future__ import annotations

from dr_llm.llm import (
    CallMode,
    Message,
    MessageRole,
    ProviderName,
    parse_llm_config,
    parse_llm_request,
)


def test_provider_name_members_behave_like_strings() -> None:
    assert ProviderName.OPENAI == str(ProviderName.OPENAI)
    assert isinstance(ProviderName.OPENAI, str)


def test_provider_name_discriminators_accept_raw_strings() -> None:
    config = parse_llm_config(
        {
            "provider": str(ProviderName.OPENAI),
            "model": "gpt-4.1-mini",
            "mode": CallMode.api,
        }
    )
    request = parse_llm_request(
        {
            "provider": str(ProviderName.OPENAI),
            "model": "gpt-4.1-mini",
            "mode": CallMode.api,
            "messages": [Message(role=MessageRole.USER, content="hi")],
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
