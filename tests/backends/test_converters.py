from __future__ import annotations

from dr_llm.backends.models import BackendRequest
from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName


def test_backend_request_round_trip_to_llm_request() -> None:
    backend_request = BackendRequest(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        messages=[Message(role="user", content="hello")],
        metadata={"trace": "1"},
        extensions={"note": "ignored-by-llm"},
    )
    llm_request = backend_request.to_llm_request()
    assert isinstance(llm_request, LlmRequest)
    assert llm_request.model == "gpt-4.1-mini"
    assert llm_request.metadata == {"trace": "1"}

    round_trip = BackendRequest.from_llm_request(
        llm_request,
        extensions=backend_request.extensions,
    )
    assert round_trip == backend_request
