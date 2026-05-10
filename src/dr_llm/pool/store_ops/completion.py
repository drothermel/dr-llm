"""Sample completion: fill in response fields for incomplete samples."""

from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import update
from sqlalchemy.sql.schema import Table

from dr_llm.pool.db import DbRuntime, SampleColumn


def complete_sample(
    runtime: DbRuntime,
    samples_table: Table,
    *,
    sample_id: str,
    response: dict[str, Any],
    finish_reason: str | None,
    attempt_count: int,
) -> bool:
    stmt = (
        update(samples_table)
        .where(
            samples_table.c[SampleColumn.SAMPLE_ID] == sample_id,
            samples_table.c[SampleColumn.RESPONSE_JSON].is_(None),
        )
        .values(
            {
                SampleColumn.RESPONSE_JSON: to_jsonable_python(response),
                SampleColumn.FINISH_REASON: finish_reason,
                SampleColumn.ATTEMPT_COUNT: attempt_count,
            }
        )
        .returning(samples_table.c[SampleColumn.SAMPLE_ID])
    )
    with runtime.begin() as conn:
        return conn.execute(stmt).scalar_one_or_none() is not None
