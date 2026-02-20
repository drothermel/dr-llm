from llm_pool.providers.utils import parse_tool_calls, parse_usage


def test_parse_usage_defaults_total() -> None:
    usage = parse_usage(prompt_tokens=10, completion_tokens=5, total_tokens=None)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


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
