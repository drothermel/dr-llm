from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm import LlmRequest, parse_llm_request


class QueuedWorkMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    work_id: str = Field(default_factory=lambda: uuid4().hex)
    request: LlmRequest
    run_id: str | None = None
    correlation_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(default=0, ge=0)

    @classmethod
    def from_payload(cls, payload: bytes) -> QueuedWorkMessage:
        raw = json.loads(payload.decode("utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("queued work payload must be a JSON object")
        if "request" not in raw:
            raise ValueError("missing 'request' field in stream payload")
        request = parse_llm_request(raw["request"])
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("queued work metadata must be a JSON object")
        kwargs: dict[str, Any] = {
            "request": request,
            "run_id": raw.get("run_id"),
            "correlation_id": raw.get("correlation_id"),
            "source": raw.get("source"),
            "metadata": metadata,
            "max_retries": raw.get("max_retries", 0),
        }
        if raw.get("work_id") is not None:
            kwargs["work_id"] = raw["work_id"]
        return cls(**kwargs)

    def json_bytes(self) -> bytes:
        payload = self.model_dump(
            mode="json", exclude_none=True, exclude_computed_fields=True
        )
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


__all__ = ["QueuedWorkMessage"]
