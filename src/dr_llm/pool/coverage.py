from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CoverageRow(BaseModel):
    """Aggregate count per unique key combination."""

    model_config = ConfigDict(frozen=True)

    key_values: dict[str, Any] = Field(default_factory=dict)
    count: int = 0
