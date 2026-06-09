from __future__ import annotations

from typing import Any

from dr_llm.backends.models import BackendRequest
from dr_llm.llm import CallMode, Message, ProviderName


def make_backend_request(**overrides: Any) -> BackendRequest:
    request = BackendRequest(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        messages=[Message(role="user", content="hello")],
    )
    if overrides:
        return request.model_copy(update=overrides)
    return request
