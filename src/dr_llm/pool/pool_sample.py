from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import Field

from dr_llm.pool.keyed_sample_base import KeyedSampleBase


class PoolSample(KeyedSampleBase):
    """A single pool sample row."""

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    sample_idx: int | None = None
    created_at: datetime | None = None

    def to_db_insert_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_idx": self.sample_idx,
            **self._base_insert_row(),
        }
