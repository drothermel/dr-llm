from __future__ import annotations

import pytest

from dr_llm.backends.errors import BackendUnsupportedFeatureError
from dr_llm.backends.validation import validate_v1_request
from tests.backends._helpers import make_backend_request


def test_validate_v1_allows_empty_extensions() -> None:
    validate_v1_request(make_backend_request())


@pytest.mark.parametrize(
    "extensions",
    [
        {"tools": []},
        {"tool_choice": "auto"},
        {"response_format": {"type": "json_object"}},
        {"stop": ["END"]},
        {"stop_sequences": ["END"]},
        {"logprobs": True},
        {"top_logprobs": 5},
        {"prompt_cache": {"enabled": True}},
        {"prompt_cache_key": "run-cache"},
        {"cache_control": {"type": "ephemeral"}},
        {"images": ["x"]},
        {"parts": [{"type": "image_url", "url": "http://example.com"}]},
    ],
)
def test_validate_v1_rejects_unsupported_extensions(
    extensions: dict[str, object],
) -> None:
    with pytest.raises(BackendUnsupportedFeatureError):
        validate_v1_request(make_backend_request(extensions=extensions))
