"""Global pool catalog table: schema persistence and discovery."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, MetaData, Table, Text, delete, select, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema

CATALOG_TABLE_NAME = "pool_catalog"


def _catalog_table(sa_metadata: MetaData) -> Table:
    return Table(
        CATALOG_TABLE_NAME,
        sa_metadata,
        Column("pool_name", Text, primary_key=True),
        Column("schema_json", JSONB, nullable=False),
        Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("now()"),
        ),
        Column(
            "updated_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("now()"),
        ),
    )


def ensure_catalog_table(runtime: DbRuntime) -> None:
    sa_metadata = MetaData()
    table = _catalog_table(sa_metadata)
    with runtime.begin() as conn:
        sa_metadata.create_all(bind=conn, tables=[table], checkfirst=True)


def upsert_schema(runtime: DbRuntime, schema: PoolSchema) -> None:
    sa_metadata = MetaData()
    table = _catalog_table(sa_metadata)
    schema_json = schema.model_dump(mode="json")
    stmt = (
        pg_insert(table)
        .values(
            pool_name=schema.name,
            schema_json=schema_json,
        )
        .on_conflict_do_update(
            index_elements=["pool_name"],
            set_={
                "schema_json": schema_json,
                "updated_at": text("now()"),
            },
        )
    )
    with runtime.begin() as conn:
        conn.execute(stmt)


def load_schema(runtime: DbRuntime, pool_name: str) -> PoolSchema | None:
    sa_metadata = MetaData()
    table = _catalog_table(sa_metadata)
    stmt = select(table.c.schema_json).where(table.c.pool_name == pool_name)
    with runtime.connect() as conn:
        row = conn.execute(stmt).scalar_one_or_none()
    if row is None:
        return None
    return PoolSchema(**row)


def load_catalog_created_at(runtime: DbRuntime, pool_name: str) -> datetime | None:
    sa_metadata = MetaData()
    table = _catalog_table(sa_metadata)
    stmt = select(table.c.created_at).where(table.c.pool_name == pool_name)
    with runtime.connect() as conn:
        return conn.execute(stmt).scalar_one_or_none()


def list_pool_names(runtime: DbRuntime) -> list[str]:
    sa_metadata = MetaData()
    table = _catalog_table(sa_metadata)
    stmt = select(table.c.pool_name).order_by(table.c.pool_name)
    with runtime.connect() as conn:
        return list(conn.execute(stmt).scalars())


def delete_catalog_entry(runtime: DbRuntime, pool_name: str) -> bool:
    sa_metadata = MetaData()
    table = _catalog_table(sa_metadata)
    stmt = (
        delete(table).where(table.c.pool_name == pool_name).returning(table.c.pool_name)
    )
    with runtime.begin() as conn:
        return conn.execute(stmt).scalar_one_or_none() is not None
