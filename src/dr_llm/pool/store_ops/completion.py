"""Sample completion: fill in response fields for incomplete samples."""

from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import exists, func, update
from sqlalchemy.sql.schema import Table

from dr_llm.pool.db import DbRuntime, LeaseColumn, SampleColumn


def complete_sample(
    runtime: DbRuntime,
    samples_table: Table,
    leases_table: Table | None = None,
    *,
    sample_id: str,
    response: dict[str, Any],
    finish_reason: str | None,
    attempt_count: int,
    lease_owner: str | None = None,
) -> bool:
    if lease_owner is not None and leases_table is None:
        raise ValueError("leases_table is required when lease_owner is provided")

    predicates = [
        samples_table.c[SampleColumn.SAMPLE_ID] == sample_id,
        samples_table.c[SampleColumn.RESPONSE_JSON].is_(None),
    ]
    if lease_owner is not None:
        assert leases_table is not None
        predicates.append(
            exists()
            .where(
                leases_table.c[LeaseColumn.SAMPLE_ID]
                == samples_table.c[SampleColumn.SAMPLE_ID],
                leases_table.c[LeaseColumn.WORKER_ID] == lease_owner,
                leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] > func.now(),
            )
            .correlate(samples_table)
        )

    stmt = (
        update(samples_table)
        .where(*predicates)
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
