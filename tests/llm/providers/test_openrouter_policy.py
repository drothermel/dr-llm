from __future__ import annotations

import csv
from pathlib import Path

from dr_llm.llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    _policies,
)


def test_openrouter_policy_covers_affordable_models_csv() -> None:
    csv_path = (
        Path(__file__).resolve().parents[3] / "info" / "affordable_models_data.csv"
    )
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    csv_models = {
        row["Model"] for row in rows if row["Source"].strip().lower() == "openrouter"
    }
    assert set(_policies()) == csv_models


def test_openrouter_policy_applies_verified_overrides() -> None:
    policies = _policies()
    assert policies["deepseek/deepseek-r1"].supports_disable is False
    assert policies["baidu/ernie-4.5-21b-a3b-thinking"].supports_disable is False
    assert policies["deepseek/deepseek-r1-0528"].supports_disable is False
    assert policies["qwen/qwen3-next-80b-a3b-thinking"].supports_disable is False
    assert policies["qwen/qwen3-30b-a3b-thinking-2507"].supports_disable is False
    assert policies["qwen/qwen3-235b-a22b-thinking-2507"].supports_disable is False
    assert policies["stepfun/step-3.5-flash"].supports_disable is False
    assert (
        policies["openai/gpt-oss-20b"].request_style
        == OpenRouterReasoningRequestStyle.EFFORT
    )
    assert (
        policies["deepseek/deepseek-chat"].request_style
        == OpenRouterReasoningRequestStyle.NONE
    )


def test_openrouter_policies_yaml_loads_and_validates() -> None:
    policies = _policies()
    assert len(policies) > 0
    for model, policy in policies.items():
        assert policy.model == model
        assert isinstance(policy.request_style, OpenRouterReasoningRequestStyle)
        assert isinstance(policy.supports_disable, bool)
