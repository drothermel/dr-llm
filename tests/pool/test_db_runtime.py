from __future__ import annotations

import pytest

from dr_llm.pool.db.runtime import _sqlalchemy_dsn, _validate_dedicated_schema


def test_sqlalchemy_dsn_adds_psycopg_driver() -> None:
    assert _sqlalchemy_dsn("postgresql://localhost/dr_llm").startswith(
        "postgresql+psycopg://"
    )


def test_sqlalchemy_dsn_preserves_existing_driver() -> None:
    assert (
        _sqlalchemy_dsn("postgresql+psycopg://localhost/dr_llm")
        == "postgresql+psycopg://localhost/dr_llm"
    )


def test_validate_dedicated_schema_requires_non_empty_value() -> None:
    with pytest.raises(ValueError, match="non-empty dedicated_schema"):
        _validate_dedicated_schema("  ")
