"""Pool metadata key-value store."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables import PoolTables


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
        stmt = pg_insert(self._tables.metadata_table).values(
            pool_name=self._schema.name,
            key=key,
            value_json=value,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                self._tables.metadata_table.c.pool_name,
                self._tables.metadata_table.c.key,
            ],
            set_={
                "value_json": stmt.excluded.value_json,
                "updated_at": func.now(),
            },
        )
        with self._runtime.begin() as conn:
            conn.execute(stmt)

    def get(self, key: str) -> dict[str, Any] | None:
        stmt = select(self._tables.metadata_table.c.value_json).where(
            self._tables.metadata_table.c.pool_name == self._schema.name,
            self._tables.metadata_table.c.key == key,
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
        stmt = (
            select(
                self._tables.metadata_table.c.key,
                self._tables.metadata_table.c.value_json,
            )
            .where(
                self._tables.metadata_table.c.pool_name == self._schema.name,
                self._tables.metadata_table.c.key.startswith(prefix, autoescape=True),
            )
            .order_by(self._tables.metadata_table.c.key)
        )
        with self._runtime.connect() as conn:
            for row in conn.execute(stmt).mappings():
                yield row["key"], row["value_json"]
