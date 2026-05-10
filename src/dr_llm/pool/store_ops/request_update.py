"""Update request payloads for incomplete pool samples."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python
from sqlalchemy import update
from sqlalchemy.sql.schema import Table

from dr_llm.pool.db import DbRuntime, SampleColumn


class RequestUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    request: dict[str, Any]
    metadata: dict[str, Any] | None = None


def update_incomplete_request(
    runtime: DbRuntime,
    samples_table: Table,
    *,
    sample_id: str,
    request: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> bool:
    values: dict[str, Any] = {
        SampleColumn.REQUEST_JSON: to_jsonable_python(request),
    }
    if metadata is not None:
        values[SampleColumn.METADATA_JSON] = to_jsonable_python(metadata)

    stmt = (
        update(samples_table)
        .where(
            samples_table.c[SampleColumn.SAMPLE_ID] == sample_id,
            samples_table.c[SampleColumn.RESPONSE_JSON].is_(None),
        )
        .values(values)
        .returning(samples_table.c[SampleColumn.SAMPLE_ID])
    )
    with runtime.begin() as conn:
        return conn.execute(stmt).scalar_one_or_none() is not None


def update_incomplete_requests(
    runtime: DbRuntime,
    samples_table: Table,
    *,
    rows: Iterable[RequestUpdate],
) -> int:
    count = 0
    for row in rows:
        if update_incomplete_request(
            runtime,
            samples_table,
            sample_id=row.sample_id,
            request=row.request,
            metadata=row.metadata,
        ):
            count += 1
    return count
