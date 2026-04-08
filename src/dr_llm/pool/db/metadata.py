"""Pool metadata key-value store."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables import PoolTables


class MetadataStore:
    """Key-value metadata store scoped to a pool."""

    def __init__(
        self,
        schema: PoolSchema,
        runtime: DbRuntime,
        tables: PoolTables,
    ) -> None:
        self._schema = schema
        self._runtime = runtime
        self._tables = tables

    def upsert_metadata(self, key: str, value_json: dict[str, Any]) -> None:
        stmt = pg_insert(self._tables.metadata_table).values(
            pool_name=self._schema.name,
            key=key,
            value_json=value_json,
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

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        stmt = select(self._tables.metadata_table.c.value_json).where(
            self._tables.metadata_table.c.pool_name == self._schema.name,
            self._tables.metadata_table.c.key == key,
        )
        with self._runtime.connect() as conn:
            raw = conn.execute(stmt).scalar_one_or_none()
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError(
                f"Expected dict from JSONB column, got {type(raw)}",
            )
        return raw
