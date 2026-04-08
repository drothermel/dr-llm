"""Shared HTTP response validation for provider responses."""

from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from dr_llm.errors import ProviderSemanticError, ProviderTransportError


def validate_http_response(
    *,
    provider_label: str,
    status_code: int,
    response_text_preview: str,
    json_error: str | None,
    response_shape_error: str | None,
) -> None:
    """Raise the appropriate provider error for a non-OK HTTP response.

    ``ProviderTransportError`` covers transient failures (5xx, 429, JSON decode
    errors). ``ProviderSemanticError`` covers client errors and shape failures
    that retrying will not resolve.
    """
    if status_code >= 500 or status_code in {408, 429}:
        raise ProviderTransportError(
            f"{provider_label} transient error status={status_code} body={response_text_preview}"
        )
    if status_code >= 400:
        raise ProviderSemanticError(
            f"{provider_label} rejected request status={status_code} body={response_text_preview}"
        )
    if json_error is not None:
        raise ProviderTransportError(
            f"{provider_label} invalid JSON response: {json_error}"
        )
    if response_shape_error is not None:
        raise ProviderSemanticError(
            f"{provider_label} response shape invalid: {response_shape_error}"
        )


T = TypeVar("T", bound=BaseModel)


def parse_http_response_body(
    response: httpx.Response,
    payload_model_cls: type[T],
) -> tuple[Any, T | None, str | None, str | None]:
    """Decode an HTTP response body and validate it against ``payload_model_cls``.

    Returns ``(decoded_json, parsed_body, json_error, shape_error)``. At most one
    of ``json_error`` and ``shape_error`` is set. On successful JSON decode,
    ``decoded_json`` is the parsed value (a dict when the API returned a JSON
    object, or a list/string/etc. when the top-level JSON is not an object—then
    ``shape_error`` is ``expected JSON object`` for debugging). When
    ``parsed_body`` is set, ``decoded_json`` is the object dict. ``parsed_body``
    is populated only on a fully successful validation.
    """
    try:
        body_raw = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, None, str(exc), None
    if not isinstance(body_raw, dict):
        return body_raw, None, None, "expected JSON object"
    try:
        parsed = payload_model_cls(**body_raw)
    except ValidationError as exc:
        return body_raw, None, None, str(exc)
    return body_raw, parsed, None, None
