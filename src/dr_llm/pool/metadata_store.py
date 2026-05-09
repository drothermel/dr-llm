"""Pool metadata key-value store."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dr_llm.pool.db import (
    DbRuntime,
    MetadataColumn,
    PoolSchema,
    PoolTables,
    PoolTableType,
)


class MetadataStore:
    """Key-value metadata store scoped to a pool.

    Keys starting with an underscore are reserved for ``dr_llm`` internal
    use (e.g. ``_schema``, populated by :meth:`PoolStore.ensure_schema`).
    Consumer-owned keys should use a prefix without a leading underscore.
    """

    def __init__(
        self,
        schema: PoolSchema,
        runtime: DbRuntime,
        tables: PoolTables,
    ) -> None:
        self._schema = schema
        self._runtime = runtime
        self._tables = tables

    def upsert(self, key: str, value: dict[str, Any]) -> None:
        metadata_table = self._tables[PoolTableType.METADATA]
        stmt = pg_insert(metadata_table).values(
            {
                MetadataColumn.POOL_NAME: self._schema.name,
                MetadataColumn.KEY: key,
                MetadataColumn.VALUE_JSON: value,
            }
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                metadata_table.c.pool_name,
                metadata_table.c.key,
            ],
            set_={
                MetadataColumn.VALUE_JSON: stmt.excluded.value_json,
                MetadataColumn.UPDATED_AT: func.now(),
            },
        )
        with self._runtime.begin() as conn:
            conn.execute(stmt)

    def get(self, key: str) -> dict[str, Any] | None:
        metadata_table = self._tables[PoolTableType.METADATA]
        stmt = select(metadata_table.c.value_json).where(
            metadata_table.c.pool_name == self._schema.name,
            metadata_table.c.key == key,
        )
        with self._runtime.connect() as conn:
            return conn.execute(stmt).scalar_one_or_none()

    def iter_prefix(self, prefix: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ``(key, value)`` pairs for metadata keys starting with ``prefix``.

        Uses an index range scan against the ``(pool_name, key)`` primary
        key, so the cost is proportional to the matched subset rather than
        the full metadata table. Pass ``""`` to iterate every key in the
        pool. Results are ordered by key for stable iteration.
        """
        metadata_table = self._tables[PoolTableType.METADATA]
        stmt = (
            select(
                metadata_table.c.key,
                metadata_table.c.value_json,
            )
            .where(
                metadata_table.c.pool_name == self._schema.name,
                metadata_table.c.key.startswith(prefix, autoescape=True),
            )
            .order_by(metadata_table.c.key)
        )
        with self._runtime.connect() as conn:
            for row in conn.execute(stmt).mappings():
                yield row[MetadataColumn.KEY], row[MetadataColumn.VALUE_JSON]
