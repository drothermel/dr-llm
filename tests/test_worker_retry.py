from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from llm_pool.session.worker import run_tool_worker
from llm_pool.types import ToolResult


@dataclass
class FakeRepository:
    claims: list[list[Any]]
    claim_calls: int = 0
    completed: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    dead_lettered: list[str] = field(default_factory=list)

    def claim_tool_calls(self, *, worker_id: str, limit: int, lease_seconds: int):  # noqa: ARG002
        idx = self.claim_calls
        self.claim_calls += 1
        if idx < len(self.claims):
            return self.claims[idx]
        return []

    def complete_tool_call(self, *, result: ToolResult) -> None:
        self.completed.append(result.tool_call_id)

    def release_tool_claim(self, *, tool_call_id: str, error_text: str | None = None) -> None:  # noqa: ARG002
        self.released.append(tool_call_id)

    def dead_letter_tool_call(self, *, tool_call_id: str, reason: str, payload: dict[str, Any] | None = None) -> str:  # noqa: ARG002
        self.dead_lettered.append(tool_call_id)
        return "dead_1"


@dataclass
class FakeExecutor:
    result: ToolResult

    def invoke(self, call):  # noqa: ANN001
        return ToolResult(
            tool_call_id=call.tool_call_id,
            ok=self.result.ok,
            result=self.result.result,
            error=self.result.error,
        )


def _call(attempt_count: int) -> Any:
    return SimpleNamespace(
        tool_call_id="tc_1",
        tool_name="echo",
        args={"x": 1},
        session_id="s_1",
        turn_id="t_1",
        attempt_count=attempt_count,
    )


def test_worker_releases_failed_call_for_retry_before_max_attempts() -> None:
    repository = FakeRepository(claims=[[_call(attempt_count=1)], []])
    executor = FakeExecutor(
        ToolResult(tool_call_id="tc_1", ok=False, error={"error_type": "RuntimeError", "message": "boom"})
    )

    stats = run_tool_worker(
        repository=repository,
        executor=executor,
        max_loops=2,
        idle_sleep_seconds=0,
        max_attempts_before_dead_letter=3,
    )

    assert stats["failed"] == 1
    assert stats["dead_lettered"] == 0
    assert repository.completed == []
    assert repository.released == ["tc_1"]
    assert repository.dead_lettered == []


def test_worker_dead_letters_when_attempt_threshold_reached() -> None:
    repository = FakeRepository(claims=[[_call(attempt_count=3)], []])
    executor = FakeExecutor(
        ToolResult(tool_call_id="tc_1", ok=False, error={"error_type": "RuntimeError", "message": "boom"})
    )

    stats = run_tool_worker(
        repository=repository,
        executor=executor,
        max_loops=2,
        idle_sleep_seconds=0,
        max_attempts_before_dead_letter=3,
    )

    assert stats["failed"] == 1
    assert stats["dead_lettered"] == 1
    assert repository.completed == []
    assert repository.released == []
    assert repository.dead_lettered == ["tc_1"]


def test_worker_completes_successful_call() -> None:
    repository = FakeRepository(claims=[[_call(attempt_count=1)], []])
    executor = FakeExecutor(ToolResult(tool_call_id="tc_1", ok=True, result={"ok": True}))

    stats = run_tool_worker(
        repository=repository,
        executor=executor,
        max_loops=2,
        idle_sleep_seconds=0,
        max_attempts_before_dead_letter=3,
    )

    assert stats["succeeded"] == 1
    assert repository.completed == ["tc_1"]
    assert repository.released == []
    assert repository.dead_lettered == []
