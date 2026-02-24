from __future__ import annotations

import json
import threading
from typing import Any
from uuid import uuid4

from llm_pool.errors import PersistenceError, SessionConflictError
from llm_pool.storage._runtime import StorageRuntime
from llm_pool.types import (
    SessionEvent,
    SessionHandle,
    SessionState,
    SessionStatus,
    SessionTurnStatus,
    ToolPolicy,
    utcnow,
)


class SessionsStore:
    def __init__(self, runtime: StorageRuntime) -> None:
        self._runtime = runtime
        self._schema_checked = False
        self._schema_lock = threading.Lock()

    def _ensure_schema(self) -> None:
        if self._schema_checked:
            return
        with self._schema_lock:
            if self._schema_checked:
                return
            self._runtime.init_schema()
            self._schema_checked = True

    def start_session(
        self,
        *,
        strategy_mode: ToolPolicy = ToolPolicy.native_preferred,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionHandle:
        self._ensure_schema()
        sid = session_id or uuid4().hex
        now = utcnow()
        with self._runtime.conn() as conn:
            try:
                row = conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        status,
                        version,
                        strategy_mode,
                        metadata_json,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        status = excluded.status,
                        strategy_mode = excluded.strategy_mode,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    RETURNING version
                    """,
                    [
                        sid,
                        SessionStatus.active.value,
                        1,
                        strategy_mode.value,
                        json.dumps(metadata or {}, ensure_ascii=True),
                        now,
                        now,
                    ],
                ).fetchone()
                conn.commit()
                version = int(row[0]) if row is not None else 1
                return SessionHandle(
                    session_id=sid,
                    status=SessionStatus.active,
                    version=version,
                    strategy_mode=strategy_mode,
                )
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to start session: {exc}") from exc

    def get_session(self, *, session_id: str) -> SessionState:
        self._ensure_schema()
        with self._runtime.conn() as conn:
            row = conn.execute(
                """
                SELECT session_id, status, version, strategy_mode, metadata_json, created_at, updated_at, last_error_text
                FROM sessions
                WHERE session_id = %s
                """,
                [session_id],
            ).fetchone()
        if row is None:
            raise PersistenceError(f"Session not found: {session_id}")
        return SessionState(
            session_id=str(row[0]),
            status=SessionStatus(str(row[1])),
            version=int(row[2]),
            strategy_mode=ToolPolicy(str(row[3])),
            metadata=row[4] if isinstance(row[4], dict) else {},
            created_at=row[5],
            updated_at=row[6],
            last_error_text=str(row[7]) if row[7] is not None else None,
        )

    def advance_session_version(self, *, session_id: str, expected_version: int) -> int:
        self._ensure_schema()
        with self._runtime.conn() as conn:
            try:
                row = conn.execute(
                    """
                    UPDATE sessions
                    SET version = version + 1,
                        updated_at = now()
                    WHERE session_id = %s
                      AND version = %s
                    RETURNING version
                    """,
                    [session_id, expected_version],
                ).fetchone()
                if row is None:
                    conn.rollback()
                    raise SessionConflictError(
                        f"Session version conflict session_id={session_id} expected={expected_version}"
                    )
                conn.commit()
                return int(row[0])
            except SessionConflictError:
                raise
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to advance session version: {exc}"
                ) from exc

    def update_session_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        last_error_text: str | None = None,
    ) -> None:
        self._ensure_schema()
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    UPDATE sessions
                    SET status = %s,
                        last_error_text = %s,
                        updated_at = now()
                    WHERE session_id = %s
                    """,
                    [status.value, last_error_text, session_id],
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to update session status: {exc}"
                ) from exc

    def create_session_turn(
        self,
        *,
        session_id: str,
        status: SessionTurnStatus = SessionTurnStatus.active,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        self._ensure_schema()
        turn_id = uuid4().hex
        with self._runtime.conn() as conn:
            try:
                session_row = conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = %s FOR UPDATE",
                    [session_id],
                ).fetchone()
                if session_row is None:
                    raise PersistenceError(
                        f"Session not found while creating turn: {session_id}"
                    )
                row = conn.execute(
                    "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM session_turns WHERE session_id = %s",
                    [session_id],
                ).fetchone()
                turn_index = int(row[0]) if row is not None else 0
                conn.execute(
                    """
                    INSERT INTO session_turns (turn_id, session_id, turn_index, status, metadata_json, created_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, now())
                    """,
                    [
                        turn_id,
                        session_id,
                        turn_index,
                        status.value,
                        json.dumps(metadata or {}, ensure_ascii=True),
                    ],
                )
                conn.commit()
                return turn_id, turn_index
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to create session turn: {exc}") from exc

    def complete_session_turn(
        self,
        *,
        turn_id: str,
        status: SessionTurnStatus,
    ) -> None:
        self._ensure_schema()
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    UPDATE session_turns
                    SET status = %s,
                        completed_at = now()
                    WHERE turn_id = %s
                    """,
                    [status.value, turn_id],
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to complete session turn: {exc}"
                ) from exc

    def append_session_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        self._ensure_schema()
        eid = event_id or uuid4().hex
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO session_events (event_id, session_id, turn_id, event_type, payload_json, created_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    [
                        eid,
                        session_id,
                        turn_id,
                        event_type,
                        json.dumps(payload, ensure_ascii=True),
                    ],
                )
                conn.commit()
                return eid
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to append session event: {exc}"
                ) from exc

    def load_session_events(self, *, session_id: str) -> list[SessionEvent]:
        self._ensure_schema()
        with self._runtime.conn() as conn:
            rows = conn.execute(
                """
                SELECT event_id, session_id, turn_id, event_type, payload_json, created_at
                FROM session_events
                WHERE session_id = %s
                ORDER BY event_seq ASC
                """,
                [session_id],
            ).fetchall()

        out: list[SessionEvent] = []
        for row in rows:
            out.append(
                SessionEvent(
                    event_id=str(row[0]),
                    session_id=str(row[1]),
                    turn_id=str(row[2]) if row[2] is not None else None,
                    event_type=str(row[3]),
                    payload=row[4] if isinstance(row[4], dict) else {},
                    created_at=row[5],
                )
            )
        return out

    def replay_session_messages(self, *, session_id: str) -> list[dict[str, Any]]:
        events = self.load_session_events(session_id=session_id)
        messages: list[dict[str, Any]] = []
        for event in events:
            if event.event_type == "message":
                message = event.payload.get("message")
                if isinstance(message, dict):
                    messages.append(message)
            elif event.event_type == "tool_result_message":
                message = event.payload.get("message")
                if isinstance(message, dict):
                    messages.append(message)
        return messages
