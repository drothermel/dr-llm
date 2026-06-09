"""v1 request validation for the backends API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dr_llm.backends.errors import BackendUnsupportedFeatureError
from dr_llm.backends.models import BackendRequest

_UNSUPPORTED_TOP_LEVEL_KEYS = frozenset(
    {
        "tools",
        "tool_choice",
        "response_format",
        "images",
        "image",
        "multimodal",
    }
)

_MULTIMODAL_TYPE_MARKERS = frozenset({"image", "image_url", "input_image"})


def validate_v1_request(request: BackendRequest) -> None:
    """Reject v1-unsupported extension features."""
    extensions = request.extensions
    if not extensions:
        return

    for key in extensions:
        if key in _UNSUPPORTED_TOP_LEVEL_KEYS:
            raise BackendUnsupportedFeatureError(
                f"v1 backends do not support extensions[{key!r}]"
            )

    if _contains_multimodal_marker(extensions):
        raise BackendUnsupportedFeatureError(
            "v1 backends do not support multimodal content in extensions"
        )


def _contains_multimodal_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        type_value = value.get("type")
        if (
            isinstance(type_value, str)
            and type_value in _MULTIMODAL_TYPE_MARKERS
        ):
            return True
        return any(
            _contains_multimodal_marker(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_multimodal_marker(item) for item in value)
    return False
