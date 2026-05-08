from __future__ import annotations

import re
from enum import StrEnum

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)


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

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "key_column_name": self.name,
                    "key_column_type": self.type.value,
                }
            ]
        )


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

    @model_validator(mode="after")
    def _validate_key_columns_nonempty(self) -> PoolSchema:
        if not self.key_columns:
            raise ValueError("PoolSchema requires at least one KeyColumn")
        return self

    @classmethod
    def from_axis_names(cls, name: str, axis_names: list[str]) -> PoolSchema:
        """Build a schema from a list of axis names, all typed as text.

        Convenience for the common case where the schema's key columns are
        the names of cross-product axes (see ``dr_llm.pool.pending.grid``).
        Each axis name becomes one ``KeyColumn`` with default ``ColumnType.text``;
        names are validated through the same regex as the regular constructor.
        """
        return cls(
            name=name,
            key_columns=[KeyColumn(name=axis_name) for axis_name in axis_names],
        )

    @computed_field
    @property
    def samples_table(self) -> str:
        return f"pool_{self.name}_samples"

    @computed_field
    @property
    def claims_table(self) -> str:
        return f"pool_{self.name}_claims"

    @computed_field
    @property
    def pending_table(self) -> str:
        return f"pool_{self.name}_pending"

    @computed_field
    @property
    def metadata_table(self) -> str:
        return f"pool_{self.name}_metadata"

    @computed_field
    @property
    def call_stats_table(self) -> str:
        return f"pool_{self.name}_call_stats"

    @property
    def key_column_names(self) -> list[str]:
        return [kc.name for kc in self.key_columns]

    def to_df(self) -> pd.DataFrame:
        from dr_llm.dataframes import single_df_row

        key_column_rows = [
            single_df_row(key_column.to_df(), source="KeyColumn")
            for key_column in self.key_columns
        ]
        return pd.DataFrame.from_records(
            [
                {
                    "pool_name": self.name,
                    "key_columns": ", ".join(
                        f"{row['key_column_name']}:{row['key_column_type']}"
                        for row in key_column_rows
                    ),
                    "samples_table": self.samples_table,
                    "claims_table": self.claims_table,
                    "pending_table": self.pending_table,
                    "metadata_table": self.metadata_table,
                    "call_stats_table": self.call_stats_table,
                }
            ]
        )
