from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class InsertResult(BaseModel):
    """Result of a bulk insert operation."""

    model_config = ConfigDict(frozen=True)

    inserted: int = 0
    skipped: int = 0
    failed: int = 0

    def __add__(self, other: InsertResult) -> InsertResult:
        return InsertResult(
            inserted=self.inserted + other.inserted,
            skipped=self.skipped + other.skipped,
            failed=self.failed + other.failed,
        )

    def __radd__(self, other: object) -> InsertResult:
        if other == 0:
            return self
        if isinstance(other, InsertResult):
            return other + self
        return NotImplemented  # type: ignore[return-value]
