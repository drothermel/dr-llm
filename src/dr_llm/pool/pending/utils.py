from typing import Any

from pydantic import BaseModel


def serialize_payload_value(value: Any) -> Any:
    """Convert common pending payload values into plain JSON-compatible shapes."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [
            item.model_dump() if isinstance(item, BaseModel) else item for item in value
        ]
    return value
