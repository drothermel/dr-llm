from __future__ import annotations

from dr_llm.llm import CallMode, LlmRequest, LlmResponse, Message, ProviderName
from dr_llm.metadata_projection import (
    EventFactMapper,
    MetadataAssertionType,
    MetadataEntityType,
    MetadataProjectionConfig,
    assertion_id,
    content_hash,
    entity_id,
)
from dr_llm.streaming_log.event_builders import (
    request_summary_from_request,
    response_summary_from_response,
)
from dr_llm.streaming_log.events import (
    EventEnvelope,
    ProducerInfo,
    ProviderRequestPreparedPayload,
    ProviderResponseReceivedPayload,
    StreamingLogEventType,
    WorkSubmittedPayload,
)


def test_metadata_identity_hashes_are_stable() -> None:
    assert entity_id("run", "run-1") == entity_id("run", "run-1")
    assert entity_id("run", "run-1") != entity_id("run", "run-2")
    assert assertion_id(
        projection_version="metadata-v1",
        assertion_type="work_submitted",
        source_idempotency_key="idem-1",
    ) == assertion_id(
        projection_version="metadata-v1",
        assertion_type="work_submitted",
        source_idempotency_key="idem-1",
    )
    assert content_hash({"b": 2, "a": 1}) == content_hash({"a": 1, "b": 2})


def test_request_and_response_summaries_are_compact() -> None:
    request = _request()
    response = _response()

    request_summary = request_summary_from_request(request)
    response_summary = response_summary_from_response(response)

    assert request_summary.provider == "openai"
    assert request_summary.message_count == 1
    assert request_summary.prompt_preview == "hello"
    assert response_summary.text_preview == "done"
    assert response_summary.finish_reason == "stop"


def test_mapper_creates_work_and_model_config_facts() -> None:
    config = MetadataProjectionConfig(database_dsn="postgresql://unused")
    request_summary = request_summary_from_request(_request())
    event = EventEnvelope(
        event_type=StreamingLogEventType.work_submitted,
        producer=ProducerInfo(
            name="test-producer", version="1", instance_id="inst"
        ),
        idempotency_key="idem-work",
        payload=WorkSubmittedPayload(
            work_id="work-1",
            run_id="run-1",
            max_retries=0,
            request_summary=request_summary,
        ),
        run_id="run-1",
        work_id="work-1",
    )

    plan = EventFactMapper(config).map_event(event)

    assert [assertion.assertion_type for assertion in plan.assertions] == [
        MetadataAssertionType.work_submitted
    ]
    entity_types = {entity.entity_type for entity in plan.entities}
    assert MetadataEntityType.run in entity_types
    assert MetadataEntityType.work in entity_types
    assert MetadataEntityType.model_config in entity_types
    assert MetadataEntityType.prompt_instance in entity_types
    assert not plan.errors


def test_mapper_creates_provider_response_facts() -> None:
    config = MetadataProjectionConfig(database_dsn="postgresql://unused")
    response_summary = response_summary_from_response(_response())
    event = EventEnvelope(
        event_type=StreamingLogEventType.provider_response_received,
        producer=ProducerInfo(name="test"),
        idempotency_key="idem-response",
        payload=ProviderResponseReceivedPayload(
            provider="openai",
            model="gpt-test",
            mode="api",
            finish_reason="stop",
            response_summary=response_summary,
        ),
        run_id="run-1",
        work_id="work-1",
        attempt_id="attempt-1",
    )

    plan = EventFactMapper(config).map_event(event)

    entity_types = {entity.entity_type for entity in plan.entities}
    assert MetadataEntityType.provider in entity_types
    assert MetadataEntityType.model in entity_types
    assert MetadataEntityType.output_result in entity_types


def test_mapper_creates_request_prepared_facts() -> None:
    config = MetadataProjectionConfig(database_dsn="postgresql://unused")
    event = EventEnvelope(
        event_type=StreamingLogEventType.provider_request_prepared,
        producer=ProducerInfo(name="test"),
        idempotency_key="idem-request",
        payload=ProviderRequestPreparedPayload(
            provider="openai",
            model="gpt-test",
            mode="api",
            request_summary=request_summary_from_request(_request()),
        ),
        run_id="run-1",
        work_id="work-1",
        attempt_id="attempt-1",
    )

    plan = EventFactMapper(config).map_event(event)

    role_names = {role.role_name for role in plan.roles}
    assert MetadataEntityType.prompt_instance in role_names
    assert MetadataEntityType.model_config in role_names


def test_prompt_instance_is_independent_of_provider_config() -> None:
    first = _request()
    second = first.model_copy(update={"model": "other-model"})
    config = MetadataProjectionConfig(database_dsn="postgresql://unused")

    first_prompt = _prompt_entity(
        EventFactMapper(config).map_event(_request_event(first))
    )
    second_prompt = _prompt_entity(
        EventFactMapper(config).map_event(_request_event(second))
    )

    assert first_prompt.entity_id == second_prompt.entity_id
    assert first_prompt.content_hash == second_prompt.content_hash
    assert first_prompt.metadata_json == second_prompt.metadata_json


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hello")],
    )


def _request_event(request: LlmRequest) -> EventEnvelope:
    return EventEnvelope(
        event_type=StreamingLogEventType.provider_request_prepared,
        producer=ProducerInfo(name="test"),
        idempotency_key=f"idem-{request.model}",
        payload=ProviderRequestPreparedPayload(
            provider=str(request.provider),
            model=request.model,
            mode=str(request.mode),
            request_summary=request_summary_from_request(request),
        ),
        run_id="run-1",
        work_id=f"work-{request.model}",
        attempt_id=f"attempt-{request.model}",
    )


def _prompt_entity(plan):
    return next(
        entity
        for entity in plan.entities
        if entity.entity_type == MetadataEntityType.prompt_instance
    )


def _response() -> LlmResponse:
    return LlmResponse(
        text="done",
        provider="openai",
        model="gpt-test",
        mode=CallMode.api,
        finish_reason="stop",
    )
