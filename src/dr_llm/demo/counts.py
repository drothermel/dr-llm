"""Small count helpers for demo progress and summaries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool import LlmPoolBackendState
from dr_llm.workers import WorkerSnapshot

type DemoCountField = Literal[
    "attempted",
    "succeeded",
    "failed",
    "had_output_text",
    "claimed",
    "completed",
    "incomplete",
    "complete",
]

ATTEMPT_SUMMARY_FIELDS: tuple[DemoCountField, ...] = (
    "attempted",
    "succeeded",
    "failed",
    "had_output_text",
)
POOL_PROGRESS_FIELDS: tuple[DemoCountField, ...] = (
    "claimed",
    "completed",
    "failed",
    "incomplete",
    "complete",
)


class DemoCounts(BaseModel):
    """Presentation-oriented count container for demo scripts."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    attempted: int = Field(default=0, ge=0)
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    had_output_text: int = Field(default=0, ge=0)
    claimed: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    incomplete: int | None = Field(default=None, ge=0)
    complete: int | None = Field(default=None, ge=0)

    def increment(self, field: DemoCountField, by: int = 1) -> None:
        """Increment one count field in place."""
        current = getattr(self, field)
        current_value = 0 if current is None else current
        setattr(self, field, current_value + by)

    def format_line(self, fields: tuple[DemoCountField, ...]) -> str:
        """Format selected fields as a stable key=value line."""
        return " ".join(
            f"{field}={self._format_value(getattr(self, field))}"
            for field in fields
        )

    def key(
        self, fields: tuple[DemoCountField, ...]
    ) -> tuple[int | None, ...]:
        """Return a comparable key for selected fields."""
        return tuple(getattr(self, field) for field in fields)

    def changed_from(
        self,
        previous: DemoCounts | None,
        fields: tuple[DemoCountField, ...],
    ) -> bool:
        """Return whether selected fields differ from a previous snapshot."""
        return previous is None or self.key(fields) != previous.key(fields)

    @classmethod
    def from_pool_snapshot(
        cls,
        snapshot: WorkerSnapshot[LlmPoolBackendState],
    ) -> DemoCounts:
        """Build display counts from a pool worker snapshot."""
        backend_state = snapshot.backend_state
        return cls(
            claimed=snapshot.counts.claimed,
            completed=snapshot.counts.completed,
            failed=snapshot.counts.failed,
            incomplete=(
                None if backend_state is None else backend_state.incomplete
            ),
            complete=None if backend_state is None else backend_state.complete,
        )

    @staticmethod
    def _format_value(value: int | None) -> str:
        if value is None:
            return "?"
        return str(value)


__all__ = [
    "ATTEMPT_SUMMARY_FIELDS",
    "DemoCountField",
    "DemoCounts",
    "POOL_PROGRESS_FIELDS",
]
