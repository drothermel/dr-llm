"""Sample insertion with auto-index allocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from dr_llm.pool.db import (
    DbRuntime,
    PoolSchema,
    PoolTables,
    PoolTableType,
    SampleColumn,
)
from dr_llm.pool.db.sql_helpers import (
    insert_keyed_samples,
    is_constraint_error,
    key_filter_clause,
    validate_key_values,
)
from dr_llm.pool.insert_result import InsertResult
from dr_llm.pool.pool_sample import PoolSample

AUTO_IDX_INSERT_RETRIES = 3


def insert_sample(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    sample: PoolSample,
    *,
    ignore_conflicts: bool = True,
) -> bool:
    result = insert_samples(
        runtime, schema, tables, [sample], ignore_conflicts=ignore_conflicts
    )
    return result.inserted == 1


def insert_samples(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    samples: Iterable[PoolSample],
    *,
    ignore_conflicts: bool = True,
) -> InsertResult:
    explicit: list[PoolSample] = []
    auto_idx: list[PoolSample] = []
    for sample in samples:
        validate_key_values(schema, sample.key_values)
        if sample.sample_idx is None:
            auto_idx.append(sample)
        else:
            explicit.append(sample)

    result = InsertResult()
    if explicit:
        result += _insert_explicit(
            runtime, tables, explicit, ignore_conflicts=ignore_conflicts
        )
    if auto_idx:
        result += _batch_insert_auto_idx(
            runtime,
            schema,
            tables,
            auto_idx,
            ignore_conflicts=ignore_conflicts,
        )
    return result


def _insert_explicit(
    runtime: DbRuntime,
    tables: PoolTables,
    samples: list[PoolSample],
    *,
    ignore_conflicts: bool,
) -> InsertResult:
    samples_table = tables[PoolTableType.SAMPLES]
    inserted = insert_keyed_samples(
        runtime,
        samples_table,
        samples_table.c.sample_id,
        [tables.sample_to_row(sample) for sample in samples],
        ignore_conflicts=ignore_conflicts,
    )
    return InsertResult(inserted=inserted, skipped=len(samples) - inserted)


def _batch_insert_auto_idx(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    samples: list[PoolSample],
    *,
    ignore_conflicts: bool = True,
) -> InsertResult:
    if not samples:
        return InsertResult()

    samples_table = tables[PoolTableType.SAMPLES]
    key_names = schema.key_column_names
    base_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = tables.sample_to_row(sample)
        row.pop(SampleColumn.SAMPLE_IDX, None)
        base_rows.append(row)

    for attempt in range(1, AUTO_IDX_INSERT_RETRIES + 1):
        try:
            with runtime.begin() as conn:
                rows = _allocate_auto_idx_rows(
                    conn,
                    schema,
                    tables,
                    base_rows=base_rows,
                    key_names=key_names,
                )
                stmt = pg_insert(samples_table)
                if ignore_conflicts:
                    stmt = stmt.on_conflict_do_nothing()
                stmt = stmt.returning(samples_table.c.sample_id)
                result = (
                    conn.execute(stmt.values(rows[0]))
                    if len(rows) == 1
                    else conn.execute(stmt, rows)
                )
                inserted = 0
                for _ in result.scalars():
                    inserted += 1
            return InsertResult(
                inserted=inserted, skipped=len(samples) - inserted
            )
        except Exception as exc:
            if is_constraint_error(exc):
                if attempt < AUTO_IDX_INSERT_RETRIES:
                    continue
                if ignore_conflicts:
                    return InsertResult(inserted=0, skipped=len(samples))
            raise
    raise AssertionError("auto-idx insert retry loop exhausted unexpectedly")


def _allocate_auto_idx_rows(
    conn: Connection,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    base_rows: list[dict[str, Any]],
    key_names: list[str],
) -> list[dict[str, Any]]:
    samples_table = tables[PoolTableType.SAMPLES]
    cell_keys = sorted(
        {tuple(row[name] for name in key_names) for row in base_rows},
        key=repr,
    )
    for cell_key in cell_keys:
        conn.execute(
            select(func.pg_advisory_xact_lock(_cell_lock_id(schema, cell_key)))
        )

    max_sample_idx_by_cell: dict[tuple[Any, ...], int] = {}
    for cell_key in cell_keys:
        key_values = dict(zip(key_names, cell_key, strict=True))
        max_sample_idx_by_cell[cell_key] = int(
            conn.execute(
                select(
                    func.coalesce(func.max(samples_table.c.sample_idx), -1)
                ).where(key_filter_clause(schema, samples_table, key_values))
            ).scalar_one()
        )

    cell_offsets: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        row = dict(base_row)
        cell_key = tuple(row[name] for name in key_names)
        cell_offsets[cell_key] = cell_offsets.get(cell_key, 0) + 1
        row[SampleColumn.SAMPLE_IDX] = (
            max_sample_idx_by_cell[cell_key] + cell_offsets[cell_key]
        )
        rows.append(row)
    return rows


def _cell_lock_id(schema: PoolSchema, cell_key: tuple[Any, ...]) -> int:
    lock_payload = json.dumps(
        {
            "pool": schema.table_name(PoolTableType.SAMPLES),
            "key_values": dict(
                zip(schema.key_column_names, cell_key, strict=True)
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(
        lock_payload.encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)
