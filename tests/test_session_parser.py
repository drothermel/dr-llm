from llm_pool.session.client import _parse_brokered_tool_calls


def test_parse_brokered_tool_calls_from_text_json() -> None:
    calls = _parse_brokered_tool_calls(
        '{"tool_calls":[{"tool_call_id":"abc","name":"search","arguments":{"q":"foo"}}]}'
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id == "abc"
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "foo"}


def test_parse_brokered_tool_calls_invalid_payload() -> None:
    assert _parse_brokered_tool_calls("not json") == []
    assert _parse_brokered_tool_calls("{}") == []
    assert _parse_brokered_tool_calls('{"tool_calls":[{"arguments":{}}]}') == []
