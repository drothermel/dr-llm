from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import Field

from dr_llm.pool.db import SampleColumn
from dr_llm.pool.keyed_sample_base import KeyedSampleBase


class PoolSample(KeyedSampleBase):
    """A single pool sample row."""

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    sample_idx: int | None = None
    created_at: datetime | None = None

    def to_db_insert_row(self) -> dict[str, Any]:
        return {
            SampleColumn.SAMPLE_ID: self.sample_id,
            SampleColumn.SAMPLE_IDX: self.sample_idx,
            **self._base_insert_row(
                payload_column=SampleColumn.PAYLOAD_JSON,
                metadata_column=SampleColumn.METADATA_JSON,
                source_run_id_column=SampleColumn.SOURCE_RUN_ID,
            ),
        }
