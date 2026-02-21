from __future__ import annotations

import os
from typing import Any

import httpx

from llm_pool.errors import ProviderSemanticError, ProviderTransportError


def api_key_from_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


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
