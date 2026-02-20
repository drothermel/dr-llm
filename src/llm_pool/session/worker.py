from __future__ import annotations

import json
import time
from uuid import uuid4

from llm_pool.storage.repository import PostgresRepository
from llm_pool.tools.executor import ToolExecutor
from llm_pool.types import ToolCallStatus, ToolInvocation


def run_tool_worker(
    *,
    repository: PostgresRepository,
    executor: ToolExecutor,
    worker_id: str | None = None,
    lease_seconds: int = 60,
    batch_size: int = 8,
    idle_sleep_seconds: float = 0.5,
    max_loops: int | None = None,
    max_attempts_before_dead_letter: int = 3,
) -> dict[str, int]:
    wid = worker_id or f"tool-worker-{uuid4().hex[:8]}"
    loops = 0
    stats = {
        "claimed": 0,
        "succeeded": 0,
        "failed": 0,
        "dead_lettered": 0,
    }
    while max_loops is None or loops < max_loops:
        loops += 1
        claimed = repository.claim_tool_calls(
            worker_id=wid,
            limit=batch_size,
            lease_seconds=lease_seconds,
        )
        if not claimed:
            time.sleep(idle_sleep_seconds)
            continue

        stats["claimed"] += len(claimed)
        for call in claimed:
            invocation = ToolInvocation(
                tool_call_id=call.tool_call_id,
                name=call.tool_name,
                arguments=call.args,
                session_id=call.session_id,
                turn_id=call.turn_id,
            )
            result = executor.invoke(invocation)
            if result.ok:
                repository.complete_tool_call(result=result)
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1
                if call.attempt_count >= max_attempts_before_dead_letter:
                    repository.dead_letter_tool_call(
                        tool_call_id=call.tool_call_id,
                        reason=str(result.error),
                        payload={
                            "worker_id": wid,
                            "status": ToolCallStatus.failed.value,
                            "error": result.error,
                        },
                    )
                    stats["dead_lettered"] += 1
                else:
                    repository.release_tool_claim(
                        tool_call_id=call.tool_call_id,
                        error_text=json.dumps(result.error, ensure_ascii=True, sort_keys=True),
                    )
        if max_loops is None:
            continue
    return stats
