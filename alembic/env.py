from __future__ import annotations

from logging.config import fileConfig
from os import getenv
from re import compile

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url
from sqlalchemy.schema import MetaData


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = MetaData()
_POOL_TABLE_RE = compile(r"^pool_[a-z][a-z0-9_]*_(samples|claims|pending|metadata)$")


def _sqlalchemy_dsn() -> str:
    raw_dsn = getenv("DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm")
    url = make_url(raw_dsn)
    if "+" in url.drivername:
        return url.render_as_string(hide_password=False)
    return url.set(drivername=f"{url.drivername}+psycopg").render_as_string(
        hide_password=False
    )


def _include_object(object_, name: str | None, type_: str, reflected, compare_to) -> bool:
    del object_, reflected, compare_to
    if type_ == "table" and name is not None and _POOL_TABLE_RE.match(name):
        return False
    return True


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
