from __future__ import annotations

from dr_llm.pool.db.runtime import _sqlalchemy_dsn


def test_sqlalchemy_dsn_adds_psycopg_driver() -> None:
    assert _sqlalchemy_dsn("postgresql://localhost/dr_llm").startswith(
        "postgresql+psycopg://"
    )


def test_sqlalchemy_dsn_preserves_existing_driver() -> None:
    assert (
        _sqlalchemy_dsn("postgresql+psycopg://localhost/dr_llm")
        == "postgresql+psycopg://localhost/dr_llm"
    )
