from __future__ import annotations

from typing import Any

from sqlalchemy import Column, MetaData, Table
from sqlalchemy.engine import Connection

from dr_llm.pool.db.names import PoolTableType
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.tables import (
    CallStatsTableDef,
    ClaimsTableDef,
    MetadataTableDef,
    PendingTableDef,
    SamplesTableDef,
    TableDef,
)


class PoolTables:
    def __init__(self, schema: PoolSchema) -> None:
        self.schema = schema
        self.sa_metadata = MetaData()
        self.defs: dict[PoolTableType, TableDef] = {
            PoolTableType.SAMPLES: SamplesTableDef(),
            PoolTableType.CLAIMS: ClaimsTableDef(),
            PoolTableType.PENDING: PendingTableDef(),
            PoolTableType.METADATA: MetadataTableDef(),
            PoolTableType.CALL_STATS: CallStatsTableDef(),
        }
        missing_table_types = set(PoolTableType).difference(self.defs)
        if missing_table_types:
            missing_values = [
                table_type.value
                for table_type in PoolTableType
                if table_type in missing_table_types
            ]
            msg = f"Missing pool table definitions: {', '.join(missing_values)}"
            raise ValueError(msg)
        self.tables: dict[PoolTableType, Table] = {
            table_type: table_def.build_table(self.schema, self.sa_metadata)
            for table_type, table_def in self.defs.items()
        }
        self._build_indexes()

    def __getitem__(self, table_type: PoolTableType) -> Table:
        return self.tables[table_type]

    @property
    def all_tables(self) -> list[Table]:
        return [self.tables[table_type] for table_type in PoolTableType]

    def key_columns(self, table_type: PoolTableType) -> list[Column[Any]]:
        if table_type not in {PoolTableType.SAMPLES, PoolTableType.PENDING}:
            msg = f"{table_type} does not have pool key columns"
            raise ValueError(msg)
        table = self[table_type]
        return [table.c[name] for name in self.schema.key_column_names]

    def select_columns(self, table_type: PoolTableType) -> list[Any]:
        return self.defs[table_type].select_columns(self[table_type], self.schema)

    def ensure_indexes(self, bind: Connection) -> None:
        """Backfill any missing named indexes for runtime-owned pool tables."""
        for table in self.all_tables:
            for index in table.indexes:
                index.create(bind=bind, checkfirst=True)

    def _build_indexes(self) -> None:
        for table_type, table_def in self.defs.items():
            table_def.build_indexes(self[table_type], self.schema)
