from pydantic import BaseModel, ConfigDict


class InsertResult(BaseModel):
    """Result of a bulk insert operation."""

    model_config = ConfigDict(frozen=True)

    inserted: int = 0
    skipped: int = 0
    failed: int = 0
