from __future__ import annotations

import re
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from dr_llm.pool.db.names import PoolTableType as _PoolTableType


class ColumnType(StrEnum):
    text = "text"
    integer = "integer"
    boolean = "boolean"
    float_ = "float"


_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def pool_table_name(pool_name: str, table_type: _PoolTableType) -> str:
    return f"pool_{pool_name}_{table_type}"


def pool_table_names(pool_name: str) -> list[str]:
    return [pool_table_name(pool_name, table_type) for table_type in _PoolTableType]


class KeyColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: ColumnType = ColumnType.text

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _VALID_NAME_RE.match(v):
            msg = (
                f"KeyColumn name must be lowercase alphanumeric with underscores, "
                f"starting with a letter; got {v!r}"
            )
            raise ValueError(msg)
        return v


class PoolSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    key_columns: list[KeyColumn]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _VALID_NAME_RE.match(v):
            msg = (
                f"Pool name must be lowercase alphanumeric with underscores, "
                f"starting with a letter; got {v!r}"
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _validate_key_columns_nonempty(self) -> PoolSchema:
        if not self.key_columns:
            raise ValueError("PoolSchema requires at least one KeyColumn")
        return self

    @classmethod
    def from_axis_names(cls, name: str, axis_names: list[str]) -> PoolSchema:
        """Build a schema from a list of axis names, all typed as text.

        Convenience for the common case where the schema's key columns are
        the names of cross-product axes (see ``dr_llm.pool.seed_grid``).
        Each axis name becomes one ``KeyColumn`` with default ``ColumnType.text``;
        names are validated through the same regex as the regular constructor.
        """
        return cls(
            name=name,
            key_columns=[KeyColumn(name=axis_name) for axis_name in axis_names],
        )

    def table_name(self, table_type: _PoolTableType) -> str:
        return pool_table_name(self.name, table_type)

    def table_names(self) -> list[str]:
        return pool_table_names(self.name)

    @property
    def key_column_names(self) -> list[str]:
        return [kc.name for kc in self.key_columns]
