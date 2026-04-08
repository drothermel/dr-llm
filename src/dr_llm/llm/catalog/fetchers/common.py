from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.coercion import as_bool, as_int, first_bool

__all__ = [
    "api_key_from_env",
    "as_bool",
    "as_int",
    "fetch_models_with_template",
    "get_json",
    "require_api_key",
    "resolve_supports_vision",
]


def resolve_supports_vision(item: dict[str, Any], *keys: str) -> bool | None:
    """Resolve a vision-support flag from the first matching boolean-ish key."""
    return first_bool(*(item.get(key) for key in keys))


def api_key_from_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def require_api_key(*, api_key: str | None, env_var: str, label: str) -> str:
    """Resolve an API key from explicit value or environment, raising on absence."""
    normalized_api_key = api_key.strip() if api_key is not None else None
    key = normalized_api_key or api_key_from_env(env_var)
    if not key:
        raise ProviderSemanticError(
            f"Missing {label} API key for catalog sync. Set {env_var}"
        )
    return key


def get_json(
    *,
    url: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url, headers=headers)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise ProviderTransportError(f"catalog fetch failed for {url}: {exc}") from exc
    if resp.status_code >= 500 or resp.status_code == 429:
        raise ProviderTransportError(
            f"catalog transient error status={resp.status_code} body={resp.text[:500]}"
        )
    if resp.status_code >= 400:
        raise ProviderSemanticError(
            f"catalog rejected status={resp.status_code} body={resp.text[:500]}"
        )
    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise ProviderTransportError(f"catalog invalid JSON for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderSemanticError(
            f"catalog invalid JSON shape for {url}: expected object"
        )
    return payload


def fetch_models_with_template(
    *,
    url: str,
    headers: dict[str, str] | None,
    items_key: str,
    item_processor: Callable[[dict[str, Any], datetime], ModelCatalogEntry | None],
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    """Fetch a JSON catalog and map each item to a `ModelCatalogEntry`.

    The processor receives the raw item dict and the shared `fetched_at` timestamp.
    Returning `None` skips the item (e.g., when the item lacks a usable ID).
    """
    payload = get_json(url=url, headers=headers)
    if items_key not in payload:
        raise ValueError(
            f"catalog payload missing items_key={items_key!r}: payload={payload!r}"
        )
    items_raw = payload[items_key]
    if not isinstance(items_raw, list):
        raise ValueError(
            f"catalog payload has non-list items_key={items_key!r}: payload={payload!r}"
        )
    items = items_raw
    now = datetime.now(UTC)
    out: list[ModelCatalogEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = item_processor(item, now)
        if entry is not None:
            out.append(entry)
    return out, payload
