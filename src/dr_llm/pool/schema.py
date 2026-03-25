from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class ColumnType(StrEnum):
    text = "text"
    integer = "integer"
    boolean = "boolean"
    float_ = "float"


_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class KeyColumn(BaseModel):
    """A single key dimension in a pool schema."""

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
    """Consumer-declared pool schema defining key dimensions and table names."""

    model_config = ConfigDict(frozen=True)

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

    @property
    def samples_table(self) -> str:
        return f"pool_{self.name}_samples"

    @property
    def claims_table(self) -> str:
        return f"pool_{self.name}_claims"

    @property
    def pending_table(self) -> str:
        return f"pool_{self.name}_pending"

    @property
    def metadata_table(self) -> str:
        return f"pool_{self.name}_metadata"

    @property
    def key_column_names(self) -> list[str]:
        return [kc.name for kc in self.key_columns]
