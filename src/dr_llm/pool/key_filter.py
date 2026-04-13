from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel


class PoolKeyEqClause(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["eq"] = "eq"
    value: Any


class PoolKeyInClause(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["in"] = "in"
    values: list[Any] = Field(min_length=1)


type PoolKeyClause = Annotated[
    PoolKeyEqClause | PoolKeyInClause, Field(discriminator="op")
]


class PoolKeyFilter(RootModel[dict[str, PoolKeyClause]]):
    model_config = ConfigDict(frozen=True)

    @classmethod
    def eq(cls, **key_values: Any) -> PoolKeyFilter:
        return cls(
            {key: PoolKeyEqClause(value=value) for key, value in key_values.items()}
        )

    @classmethod
    def in_(cls, **key_values: list[Any]) -> PoolKeyFilter:
        return cls(
            {key: PoolKeyInClause(values=values) for key, values in key_values.items()}
        )
