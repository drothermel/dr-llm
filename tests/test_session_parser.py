from llm_pool.session.models import parse_brokered_tool_calls


def test_parse_brokered_tool_calls_from_text_json() -> None:
    calls = parse_brokered_tool_calls(
        '{"tool_calls":[{"tool_call_id":"abc","name":"search","arguments":{"q":"foo"}}]}'
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id == "abc"
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "foo"}


def test_parse_brokered_tool_calls_invalid_payload() -> None:
    assert parse_brokered_tool_calls("not json") == []
    assert parse_brokered_tool_calls("{}") == []
    assert parse_brokered_tool_calls('{"tool_calls":[{"arguments":{}}]}') == []


def test_parse_brokered_tool_calls_generates_uuid_when_id_missing() -> None:
    calls = parse_brokered_tool_calls(
        '{"tool_calls":[{"name":"search","arguments":{"q":"foo"}}]}'
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id.startswith("brokered_call_")
    assert len(calls[0].tool_call_id) == len("brokered_call_") + 32
    assert calls[0].name == "search"
