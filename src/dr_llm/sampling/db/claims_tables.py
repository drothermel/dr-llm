"""Manages consumer-scoped claims tables for a pool."""

from __future__ import annotations

from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Connection

from dr_llm.pool.db import DbRuntime
from dr_llm.sampling.db.names import claims_table_name
from dr_llm.sampling.db.tables import ClaimsTableDef


class ClaimsTables:
    """Builds and manages a consumer-scoped claims table.

    Each consumer (e.g., an experimental sweep) gets its own claims table
    named ``pool_{pool_name}_claims_{consumer_id}``.
    """

    def __init__(self, pool_name: str, consumer_id: str) -> None:
        self.pool_name = pool_name
        self.consumer_id = consumer_id
        self.sa_metadata = MetaData()
        self._table_def = ClaimsTableDef()
        self._table_name = claims_table_name(pool_name, consumer_id)
        self.claims_table: Table = self._table_def.build_table(
            self._table_name, self.sa_metadata
        )
        self._table_def.build_indexes(self.claims_table)

    @property
    def table_name(self) -> str:
        return self._table_name

    def ensure_table(self, runtime: DbRuntime) -> None:
        with runtime.begin() as conn:
            self.sa_metadata.create_all(
                bind=conn,
                tables=[self.claims_table],
                checkfirst=True,
            )
            self._ensure_indexes(conn)

    def drop_table(self, runtime: DbRuntime) -> None:
        with runtime.begin() as conn:
            self.claims_table.drop(bind=conn, checkfirst=True)

    def _ensure_indexes(self, conn: Connection) -> None:
        for index in self.claims_table.indexes:
            index.create(bind=conn, checkfirst=True)
