from __future__ import annotations

from dr_llm.pool.db.dsn import sqlalchemy_dsn


def test_sqlalchemy_dsn_adds_psycopg_driver() -> None:
    assert sqlalchemy_dsn("postgresql://localhost/dr_llm").startswith(
        "postgresql+psycopg://"
    )


def test_sqlalchemy_dsn_preserves_existing_driver() -> None:
    assert (
        sqlalchemy_dsn("postgresql+psycopg://localhost/dr_llm")
        == "postgresql+psycopg://localhost/dr_llm"
    )


def test_sqlalchemy_dsn_postgres_alias_matches_postgresql() -> None:
    """Like test_sqlalchemy_dsn_adds_psycopg_driver, but for the postgres:// alias."""
    assert sqlalchemy_dsn("postgres://localhost/dr_llm").startswith(
        "postgresql+psycopg://"
    )


def test_sqlalchemy_dsn_preserves_other_drivers() -> None:
    """Like test_sqlalchemy_dsn_preserves_existing_driver, for non-psycopg drivers."""
    dsn = "postgresql+asyncpg://localhost/dr_llm"
    assert sqlalchemy_dsn(dsn) == dsn
