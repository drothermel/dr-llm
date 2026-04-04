from __future__ import annotations

import csv
from pathlib import Path

from dr_llm.providers.openrouter.policy import (
    OPENROUTER_MODEL_POLICIES,
    OpenRouterReasoningRequestStyle,
)


def test_openrouter_policy_covers_affordable_models_csv() -> None:
    rows = list(
        csv.DictReader(
            Path("info/affordable_models_data.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    csv_models = {
        row["Model"]
        for row in rows
        if row["Source"].strip().lower() == "openrouter"
    }
    assert set(OPENROUTER_MODEL_POLICIES) == csv_models


def test_openrouter_policy_applies_verified_overrides() -> None:
    assert (
        OPENROUTER_MODEL_POLICIES["deepseek/deepseek-r1"].supports_disable is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["baidu/ernie-4.5-21b-a3b-thinking"].supports_disable
        is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["deepseek/deepseek-r1-0528"].supports_disable
        is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["qwen/qwen3-next-80b-a3b-thinking"].supports_disable
        is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["qwen/qwen3-30b-a3b-thinking-2507"].supports_disable
        is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["qwen/qwen3-235b-a22b-thinking-2507"].supports_disable
        is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["stepfun/step-3.5-flash"].supports_disable
        is False
    )
    assert (
        OPENROUTER_MODEL_POLICIES["openai/gpt-oss-20b"].request_style
        == OpenRouterReasoningRequestStyle.EFFORT
    )
    assert (
        OPENROUTER_MODEL_POLICIES["deepseek/deepseek-chat"].request_style
        == OpenRouterReasoningRequestStyle.NONE
    )
