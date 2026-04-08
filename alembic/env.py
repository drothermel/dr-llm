from __future__ import annotations

import re
from logging.config import fileConfig
from os import getenv

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.schema import MetaData

from dr_llm.pool.db.dsn import sqlalchemy_dsn

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = MetaData()
# Pool tables are named per PoolSchema at runtime, so Alembic intentionally does
# not own them yet. Runtime initialization remains the source of truth until the
# pool schema design moves away from dynamic per-pool physical tables.
_POOL_TABLE_RE = re.compile(r"^pool_[a-z][a-z0-9_]*_(samples|claims|pending|metadata)$")


def _sqlalchemy_dsn() -> str:
    return sqlalchemy_dsn(
        getenv("DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm")
    )


def _include_object(
    object_, name: str | None, type_: str, reflected, compare_to
) -> bool:
    del object_, reflected, compare_to
    return not (type_ == "table" and name is not None and _POOL_TABLE_RE.match(name))


def run_migrations_offline() -> None:
    context.configure(
        url=_sqlalchemy_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=_include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _sqlalchemy_dsn()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=_include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
