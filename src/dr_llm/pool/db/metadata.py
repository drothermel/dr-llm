"""Pool metadata key-value store."""

from __future__ import annotations

import json
from typing import Any

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import q


class MetadataStore:
    """Key-value metadata store scoped to a pool."""

    def __init__(self, schema: PoolSchema, runtime: DbRuntime) -> None:
        self._schema = schema
        self._runtime = runtime

    def upsert_metadata(self, key: str, value_json: dict[str, Any]) -> None:
        tbl = self._schema.metadata_table
        with self._runtime.conn() as conn:
            conn.execute(
                q(
                    f"INSERT INTO {tbl} (pool_name, key, value_json) "
                    f"VALUES (%s, %s, %s) "
                    f"ON CONFLICT (pool_name, key) DO UPDATE SET "
                    f"value_json = EXCLUDED.value_json, updated_at = now()"
                ),
                [self._schema.name, key, json.dumps(value_json, default=str)],
            )
            conn.commit()

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        tbl = self._schema.metadata_table
        with self._runtime.conn() as conn:
            row = conn.execute(
                q(f"SELECT value_json FROM {tbl} WHERE pool_name = %s AND key = %s"),
                [self._schema.name, key],
            ).fetchone()
            if row is None:
                return None
            raw = row[0]
            return raw if isinstance(raw, dict) else json.loads(raw)
