from __future__ import annotations

from typing import Any, cast

from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.streaming_log.client import StreamingLogClient
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.events import StreamingLogEventType
from dr_llm.streaming_log.work import QueuedWorkMessage


class FakeObjectResult:
    def __init__(self, data: bytes) -> None:
        self.data = data


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def get(self, name: str):
        from nats.js.errors import ObjectNotFoundError

        if name not in self.objects:
            raise ObjectNotFoundError
        return FakeObjectResult(self.objects[name])

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data


class FakeJetStream:
    def __init__(self) -> None:
        self.store = FakeObjectStore()
        self.published: list[tuple[str, bytes, str | None]] = []

    async def object_store(self, bucket: str) -> FakeObjectStore:
        assert bucket == "DRLLM_PAYLOADS"
        return self.store

    async def publish(
        self, subject: str, payload: bytes, *, stream: str | None = None
    ) -> object:
        self.published.append((subject, payload, stream))
        return object()


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def test_submit_work_publishes_event_before_work_message() -> None:
    client = StreamingLogClient(StreamingLogConfig())
    fake_js = FakeJetStream()
    client._js = cast(Any, fake_js)

    work = QueuedWorkMessage(work_id="work-1", request=_request())

    import asyncio

    event = asyncio.run(client.submit_work(work))

    assert event.event_type is StreamingLogEventType.work_submitted
    assert fake_js.published[0][0] == "drllm.events.work_submitted"
    assert fake_js.published[0][2] == "DRLLM_EVENTS"
    assert fake_js.published[1][0] == "drllm.work.llm"
    assert fake_js.published[1][2] == "DRLLM_WORK"
    assert fake_js.store.objects
