from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PayloadEncoding = Literal["utf-8", "binary"]
PayloadCompression = Literal["none"]


class PayloadRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    object_key: str
    sha256: str
    size_bytes: int = Field(ge=0)
    content_type: str
    encoding: PayloadEncoding
    compression: PayloadCompression = "none"


class PreparedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    data: bytes
    sha256: str
    object_key: str
    size_bytes: int = Field(ge=0)
    content_type: str
    encoding: PayloadEncoding
    compression: PayloadCompression = "none"

    def ref(self) -> PayloadRef:
        return PayloadRef(
            role=self.role,
            object_key=self.object_key,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            content_type=self.content_type,
            encoding=self.encoding,
            compression=self.compression,
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_key_for_sha256(digest: str) -> str:
    if len(digest) != 64:
        raise ValueError("sha256 digest must be 64 hex characters")
    int(digest, 16)
    return f"sha256/{digest[:2]}/{digest}"


def serialize_json_payload(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def prepare_json_payload(role: str, value: Any) -> PreparedPayload:
    data = serialize_json_payload(value)
    return prepare_payload(
        role=role,
        data=data,
        content_type="application/json",
        encoding="utf-8",
    )


def prepare_text_payload(role: str, value: str) -> PreparedPayload:
    return prepare_payload(
        role=role,
        data=value.encode("utf-8"),
        content_type="text/plain",
        encoding="utf-8",
    )


def prepare_payload(
    *,
    role: str,
    data: bytes,
    content_type: str,
    encoding: PayloadEncoding,
    compression: PayloadCompression = "none",
) -> PreparedPayload:
    digest = sha256_bytes(data)
    return PreparedPayload(
        role=role,
        data=data,
        sha256=digest,
        object_key=object_key_for_sha256(digest),
        size_bytes=len(data),
        content_type=content_type,
        encoding=encoding,
        compression=compression,
    )


__all__ = [
    "PayloadRef",
    "PreparedPayload",
    "object_key_for_sha256",
    "prepare_json_payload",
    "prepare_payload",
    "prepare_text_payload",
    "serialize_json_payload",
    "sha256_bytes",
]
