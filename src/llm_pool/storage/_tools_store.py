from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from llm_pool.errors import PersistenceError
from llm_pool.storage._runtime import StorageRuntime
from llm_pool.types import ToolCallRecord, ToolCallStatus, ToolResult


class ToolsStore:
    def __init__(self, runtime: StorageRuntime) -> None:
        self._runtime = runtime

    def enqueue_tool_call(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        idempotency_key: str,
        turn_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        self._runtime.init_schema()
        tcid = tool_call_id or uuid4().hex
        with self._runtime.conn() as conn:
            try:
                row = conn.execute(
                    """
                    INSERT INTO tool_calls (
                        tool_call_id,
                        session_id,
                        turn_id,
                        idempotency_key,
                        tool_name,
                        status,
                        args_json,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (idempotency_key)
                    DO UPDATE SET
                        tool_name = excluded.tool_name,
                        args_json = excluded.args_json
                    RETURNING tool_call_id
                    """,
                    [
                        tcid,
                        session_id,
                        turn_id,
                        idempotency_key,
                        tool_name,
                        ToolCallStatus.pending.value,
                        json.dumps(args, ensure_ascii=True),
                    ],
                ).fetchone()
                conn.commit()
                return str(row[0]) if row is not None else tcid
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to enqueue tool call: {exc}") from exc

    def claim_tool_calls(
        self, *, worker_id: str, limit: int, lease_seconds: int
    ) -> list[ToolCallRecord]:
        self._runtime.init_schema()
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be >= 1")

        with self._runtime.conn() as conn:
            try:
                rows = conn.execute(
                    """
                    WITH candidates AS (
                        SELECT tool_call_id
                        FROM tool_calls
                        WHERE (
                            status = %s
                            OR (status = %s AND lease_expires_at < now())
                        )
                        ORDER BY created_at ASC
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE tool_calls tc
                    SET
                        status = %s,
                        worker_id = %s,
                        claimed_at = now(),
                        lease_expires_at = now() + make_interval(secs => %s),
                        attempt_count = attempt_count + 1
                    FROM candidates
                    WHERE tc.tool_call_id = candidates.tool_call_id
                    RETURNING
                        tc.tool_call_id,
                        tc.session_id,
                        tc.turn_id,
                        tc.idempotency_key,
                        tc.tool_name,
                        tc.status,
                        tc.args_json,
                        tc.attempt_count,
                        tc.worker_id,
                        tc.created_at,
                        tc.claimed_at,
                        tc.lease_expires_at
                    """,
                    [
                        ToolCallStatus.pending.value,
                        ToolCallStatus.claimed.value,
                        int(limit),
                        ToolCallStatus.claimed.value,
                        worker_id,
                        int(lease_seconds),
                    ],
                ).fetchall()
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to claim tool calls: {exc}") from exc

        claimed: list[ToolCallRecord] = []
        for row in rows:
            claimed.append(
                ToolCallRecord(
                    tool_call_id=str(row[0]),
                    session_id=str(row[1]),
                    turn_id=str(row[2]) if row[2] is not None else None,
                    idempotency_key=str(row[3]),
                    tool_name=str(row[4]),
                    status=ToolCallStatus(str(row[5])),
                    args=row[6] if isinstance(row[6], dict) else {},
                    attempt_count=int(row[7]),
                    worker_id=str(row[8]) if row[8] is not None else None,
                    created_at=row[9],
                    claimed_at=row[10],
                    lease_expires_at=row[11],
                )
            )
        return claimed

    def renew_tool_lease(
        self, *, tool_call_id: str, worker_id: str, lease_seconds: int
    ) -> bool:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            try:
                row = conn.execute(
                    """
                    UPDATE tool_calls
                    SET lease_expires_at = now() + make_interval(secs => %s)
                    WHERE tool_call_id = %s
                      AND worker_id = %s
                      AND status = %s
                    RETURNING tool_call_id
                    """,
                    [
                        int(lease_seconds),
                        tool_call_id,
                        worker_id,
                        ToolCallStatus.claimed.value,
                    ],
                ).fetchone()
                conn.commit()
                return row is not None
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to renew tool lease: {exc}") from exc

    def release_tool_claim(
        self, *, tool_call_id: str, error_text: str | None = None
    ) -> None:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    UPDATE tool_calls
                    SET status = %s,
                        worker_id = NULL,
                        claimed_at = NULL,
                        lease_expires_at = NULL,
                        last_error_text = %s
                    WHERE tool_call_id = %s
                    """,
                    [ToolCallStatus.pending.value, error_text, tool_call_id],
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to release tool claim: {exc}") from exc

    def complete_tool_call(self, *, result: ToolResult) -> None:
        self._runtime.init_schema()
        status = ToolCallStatus.succeeded if result.ok else ToolCallStatus.failed
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    UPDATE tool_calls
                    SET status = %s,
                        lease_expires_at = NULL,
                        last_error_text = %s
                    WHERE tool_call_id = %s
                    """,
                    [
                        status.value,
                        json.dumps(
                            result.error.model_dump(
                                mode="json",
                                exclude_none=True,
                                exclude_computed_fields=True,
                            ),
                            ensure_ascii=True,
                        )
                        if result.error
                        else None,
                        result.tool_call_id,
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO tool_results (tool_call_id, result_json, error_json, completed_at)
                    VALUES (%s, %s::jsonb, %s::jsonb, now())
                    ON CONFLICT (tool_call_id)
                    DO UPDATE SET
                        result_json = excluded.result_json,
                        error_json = excluded.error_json,
                        completed_at = excluded.completed_at
                    """,
                    [
                        result.tool_call_id,
                        json.dumps(result.result, ensure_ascii=True)
                        if result.result is not None
                        else None,
                        json.dumps(
                            result.error.model_dump(
                                mode="json",
                                exclude_none=True,
                                exclude_computed_fields=True,
                            ),
                            ensure_ascii=True,
                        )
                        if result.error is not None
                        else None,
                    ],
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to complete tool call: {exc}") from exc

    def dead_letter_tool_call(
        self,
        *,
        tool_call_id: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        self._runtime.init_schema()
        dead_id = uuid4().hex
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    UPDATE tool_calls
                    SET status = %s,
                        lease_expires_at = NULL,
                        last_error_text = %s
                    WHERE tool_call_id = %s
                    """,
                    [ToolCallStatus.dead_letter.value, reason, tool_call_id],
                )
                conn.execute(
                    """
                    INSERT INTO tool_call_dead_letters (dead_letter_id, tool_call_id, reason, payload_json, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, now())
                    """,
                    [
                        dead_id,
                        tool_call_id,
                        reason,
                        json.dumps(payload or {}, ensure_ascii=True),
                    ],
                )
                conn.commit()
                return dead_id
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to dead-letter tool call: {exc}"
                ) from exc
