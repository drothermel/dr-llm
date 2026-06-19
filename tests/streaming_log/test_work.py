from __future__ import annotations

import json

from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.streaming_log.work import QueuedWorkMessage


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def test_queued_work_message_round_trips_json_bytes() -> None:
    work = QueuedWorkMessage(
        work_id="work-1",
        request=_request(),
        run_id="run-1",
        max_retries=2,
        metadata={"kind": "test"},
    )

    parsed = QueuedWorkMessage.from_payload(work.json_bytes())

    assert parsed == work


def test_queued_work_message_from_payload_generates_missing_work_id() -> None:
    payload = {
        "request": _request().model_dump(mode="json"),
        "metadata": {"kind": "test"},
    }

    parsed = QueuedWorkMessage.from_payload(
        json.dumps(payload).encode("utf-8")
    )

    assert parsed.work_id
    assert parsed.request == _request()
