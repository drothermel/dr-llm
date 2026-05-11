from __future__ import annotations

from dr_llm.llm.providers.impls.openrouter.families import (
    OPENROUTER_FAMILIES,
    OpenRouterControlRequestStyle,
)


def test_openrouter_policy_applies_verified_overrides() -> None:
    policies = OPENROUTER_FAMILIES.policies
    assert policies["deepseek/deepseek-r1"].supports_disable is False
    assert (
        policies["baidu/ernie-4.5-21b-a3b-thinking"].supports_disable is False
    )
    assert policies["deepseek/deepseek-r1-0528"].supports_disable is False
    assert (
        policies["qwen/qwen3-next-80b-a3b-thinking"].supports_disable is False
    )
    assert (
        policies["qwen/qwen3-30b-a3b-thinking-2507"].supports_disable is False
    )
    assert (
        policies["qwen/qwen3-235b-a22b-thinking-2507"].supports_disable
        is False
    )
    assert policies["stepfun/step-3.5-flash"].supports_disable is False
    assert (
        policies["openai/gpt-oss-20b"].request_style
        == OpenRouterControlRequestStyle.EFFORT
    )
    assert (
        policies["openai/gpt-5-nano"].request_style
        == OpenRouterControlRequestStyle.EFFORT
    )
    assert policies["openai/gpt-5-nano"].allowed_efforts == (
        "low",
        "medium",
        "high",
    )
    assert (
        policies["openai/gpt-5.4-nano"].request_style
        == OpenRouterControlRequestStyle.EFFORT
    )
    assert policies["openai/gpt-5.4-nano"].allowed_efforts == (
        "low",
        "medium",
        "high",
    )
    assert (
        policies["deepseek/deepseek-chat"].request_style
        == OpenRouterControlRequestStyle.NONE
    )


def test_openrouter_policies_yaml_loads_and_validates() -> None:
    policies = OPENROUTER_FAMILIES.policies
    assert len(policies) > 0
    for model, policy in policies.items():
        assert policy.model == model
        assert isinstance(policy.request_style, OpenRouterControlRequestStyle)
        assert isinstance(policy.supports_disable, bool)
