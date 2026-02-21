from llm_pool.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
    parse_tool_calls,
    parse_usage,
    to_openai_messages,
)
from llm_pool.types import Message, ModelToolCall


def test_parse_usage_defaults_total() -> None:
    usage = parse_usage(prompt_tokens=10, completion_tokens=5, total_tokens=None)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.reasoning_tokens == 0


def test_parse_usage_includes_reasoning_tokens() -> None:
    usage = parse_usage(
        prompt_tokens=10, completion_tokens=5, total_tokens=20, reasoning_tokens=7
    )
    assert usage.reasoning_tokens == 7


def test_parse_tool_calls_parses_json_arguments() -> None:
    calls = parse_tool_calls(
        [
            {
                "id": "abc",
                "function": {
                    "name": "lookup",
                    "arguments": '{"query":"x"}',
                },
            }
        ]
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id == "abc"
    assert calls[0].name == "lookup"
    assert calls[0].arguments == {"query": "x"}


def test_parse_tool_calls_skips_missing_name() -> None:
    calls = parse_tool_calls([{"id": "abc", "function": {"arguments": "{}"}}])
    assert calls == []


def test_parse_reasoning_tokens_nested_details() -> None:
    usage_raw = {"completion_tokens_details": {"reasoning_tokens": 13}}
    assert parse_reasoning_tokens(usage_raw) == 13


def test_parse_reasoning_fields() -> None:
    reasoning, reasoning_details = parse_reasoning(
        {
            "reasoning": "thought process",
            "reasoning_details": [{"type": "reasoning.text", "text": "step"}],
        }
    )
    assert reasoning == "thought process"
    assert reasoning_details == [{"type": "reasoning.text", "text": "step"}]


def test_parse_cost_info() -> None:
    cost = parse_cost_info(
        {
            "usage": {
                "cost": 0.0123,
                "prompt_cost": 0.004,
                "completion_cost": 0.0083,
                "currency": "USD",
            }
        }
    )
    assert cost is not None
    assert cost.total_cost_usd == 0.0123
    assert cost.prompt_cost_usd == 0.004
    assert cost.completion_cost_usd == 0.0083
    assert cost.currency == "USD"


def test_to_openai_messages_includes_tool_call_id() -> None:
    payload = to_openai_messages(
        [
            Message(role="tool", content='{"ok":true}', tool_call_id="tc_123"),
        ]
    )
    assert payload[0]["tool_call_id"] == "tc_123"


def test_to_openai_messages_includes_assistant_tool_calls() -> None:
    payload = to_openai_messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ModelToolCall(
                        tool_call_id="tc_1", name="lookup", arguments={"q": "abc"}
                    )
                ],
            )
        ]
    )
    assert payload[0]["tool_calls"][0]["id"] == "tc_1"
    assert payload[0]["tool_calls"][0]["function"]["name"] == "lookup"
