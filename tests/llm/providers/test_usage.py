from __future__ import annotations

from dr_llm.llm.providers.usage import CostInfo, TokenUsage, parse_reasoning


def test_token_usage_defaults_total():
    usage = TokenUsage.from_raw(prompt_tokens=10, completion_tokens=5, total_tokens=None)
    assert usage.total_tokens == 15
    assert usage.reasoning_tokens == 0


def test_token_usage_includes_reasoning_tokens():
    usage = TokenUsage.from_raw(prompt_tokens=10, completion_tokens=5, reasoning_tokens=7)
    assert usage.reasoning_tokens == 7


def test_extract_reasoning_tokens_nested_details():
    raw = {"completion_tokens_details": {"reasoning_tokens": 13}}
    assert TokenUsage.extract_reasoning_tokens(raw) == 13


def test_extract_reasoning_tokens_output_details():
    raw = {"output_tokens_details": {"reasoning_tokens": 7}}
    assert TokenUsage.extract_reasoning_tokens(raw) == 7


def test_extract_reasoning_tokens_missing_returns_zero():
    assert TokenUsage.extract_reasoning_tokens({}) == 0


def test_parse_reasoning_extracts_text_and_details():
    details = [{"type": "thinking", "text": "step 1"}]
    msg = {"reasoning": "some reasoning", "reasoning_details": details}
    text, extracted_details = parse_reasoning(msg)
    assert text == "some reasoning"
    assert extracted_details == details


def test_parse_reasoning_extracts_thinking_content_blocks():
    details = [{"type": "thinking", "thinking": "step 1"}]
    msg = {"content": details}
    text, extracted_details = parse_reasoning(msg)
    assert text == "step 1"
    assert extracted_details is None


def test_parse_reasoning_preserves_reasoning_details_when_thinking_blocks_present():
    reasoning_details = [{"type": "reasoning.text", "text": "provider detail"}]
    content = [{"type": "thinking", "thinking": "step 1"}]
    msg = {"reasoning_details": reasoning_details, "content": content}
    text, extracted_details = parse_reasoning(msg)
    assert text == "step 1"
    assert extracted_details == reasoning_details


def test_cost_info_from_raw_parses_usage_block():
    body = {
        "usage": {
            "cost": 0.05,
            "prompt_cost": 0.02,
            "completion_cost": 0.03,
        }
    }
    info = CostInfo.from_raw(body)
    assert info is not None
    assert info.total_cost_usd == 0.05
    assert info.prompt_cost_usd == 0.02
    assert info.completion_cost_usd == 0.03


def test_cost_info_from_raw_returns_none_when_no_costs():
    assert CostInfo.from_raw({"usage": {"prompt_tokens": 100}}) is None


def test_cost_info_from_raw_defaults_non_string_currency():
    body = {"usage": {"cost": 0.01, "currency": 0}}
    info = CostInfo.from_raw(body)
    assert info is not None
    assert info.currency == "USD"
