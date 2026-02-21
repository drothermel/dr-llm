from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from hashlib import sha256
from os import getenv
from pathlib import Path
from typing import Any, Generator, LiteralString, cast
from uuid import uuid4

import psycopg
from psycopg import errors, sql
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_pool.errors import (
    PersistenceError,
    SessionConflictError,
    TransientPersistenceError,
)
from llm_pool.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    RecordedCall,
    RunStatus,
    SessionEvent,
    SessionHandle,
    SessionState,
    SessionStatus,
    SessionTurnStatus,
    ToolCallRecord,
    ToolCallStatus,
    ToolPolicy,
    ToolResult,
    utcnow,
)


_SCHEMA_PATH = Path(__file__).with_name("schema_bootstrap_pg.sql")


class StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dsn: str = Field(
        default_factory=lambda: getenv(
            "LLM_POOL_DATABASE_URL", "postgresql://localhost/llm_pool"
        )
    )
    min_pool_size: int = 4
    max_pool_size: int = 64
    statement_timeout_ms: int | None = None
    application_name: str = "llm_pool"


def _is_retryable_db_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            errors.DeadlockDetected,
            errors.SerializationFailure,
        ),
    ):
        return True
    if isinstance(exc, TransientPersistenceError):
        return True
    return False


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


class PostgresRepository:
    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self._pool = ConnectionPool(
            self.config.dsn,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            open=True,
        )
        self._schema_lock = threading.Lock()
        self._schema_initialized = False

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _conn(self) -> Generator[psycopg.Connection[tuple[Any, ...]], None, None]:
        with self._pool.connection() as conn:
            if self.config.statement_timeout_ms is not None:
                conn.execute(
                    "SET statement_timeout = %s",
                    [int(self.config.statement_timeout_ms)],
                )
            yield conn

    def init_schema(self) -> None:
        if self._schema_initialized:
            return
        with self._schema_lock:
            if self._schema_initialized:
                return
            schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            with self._conn() as conn:
                conn.execute(sql.SQL(cast(LiteralString, schema_sql)))
                conn.commit()
            self._schema_initialized = True

    @retry(
        retry=retry_if_exception(_is_retryable_db_error),
        wait=wait_exponential_jitter(initial=0.05, max=2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def start_run(
        self,
        *,
        run_type: str = "generic",
        status: RunStatus = RunStatus.running,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        self.init_schema()
        resolved_run_id = run_id or uuid4().hex
        with self._conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO runs (run_id, run_type, status, metadata_json, started_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (run_id)
                    DO UPDATE SET run_type = excluded.run_type,
                                  status = excluded.status,
                                  metadata_json = excluded.metadata_json
                    """,
                    [
                        resolved_run_id,
                        run_type,
                        status.value,
                        json.dumps(metadata or {}, ensure_ascii=True),
                    ],
                )
                conn.commit()
                return resolved_run_id
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to start run: {exc}") from exc

    def upsert_run_parameters(self, *, run_id: str, parameters: dict[str, Any]) -> int:
        self.init_schema()
        if not parameters:
            return 0
        with self._conn() as conn:
            written = 0
            try:
                for key, value in parameters.items():
                    conn.execute(
                        """
                        INSERT INTO run_parameters (run_id, param_key, param_value, created_at)
                        VALUES (%s, %s, %s, now())
                        ON CONFLICT (run_id, param_key)
                        DO UPDATE SET param_value = excluded.param_value,
                                      created_at = excluded.created_at
                        """,
                        [
                            run_id,
                            str(key),
                            json.dumps(value, ensure_ascii=True, sort_keys=True),
                        ],
                    )
                    written += 1
                conn.commit()
                return written
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to upsert run parameters: {exc}"
                ) from exc

    def finish_run(
        self,
        *,
        run_id: str,
        status: RunStatus,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.init_schema()
        with self._conn() as conn:
            try:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = %s,
                        metadata_json = COALESCE(%s::jsonb, metadata_json),
                        finished_at = now()
                    WHERE run_id = %s
                    """,
                    [
                        status.value,
                        json.dumps(metadata, ensure_ascii=True)
                        if metadata is not None
                        else None,
                        run_id,
                    ],
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to finish run: {exc}") from exc

    def record_call(
        self,
        *,
        request: LlmRequest,
        response: LlmResponse | None = None,
        run_id: str | None = None,
        status: str | None = None,
        mode: CallMode | str | None = None,
        error_text: str | None = None,
        external_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> str:
        self.init_schema()
        resolved_call_id = call_id or uuid4().hex
        request_payload = request.model_dump(mode="json", exclude_computed_fields=True)
        response_payload = (
            response.model_dump(mode="json", exclude_computed_fields=True)
            if response is not None
            else None
        )
        request_hash = _hash_payload(request_payload)
        response_hash = (
            _hash_payload(response_payload) if response_payload is not None else None
        )
        resolved_status = status or (
            "success" if response is not None and error_text is None else "failed"
        )
        if mode is not None:
            resolved_mode = mode.value if isinstance(mode, CallMode) else str(mode)
        elif response is not None:
            resolved_mode = response.mode.value
        else:
            resolved_mode = "api"
        latency_ms = int(response.latency_ms) if response is not None else 0

        with self._conn() as conn:
            try:
                row = conn.execute(
                    """
                    INSERT INTO llm_calls (
                        call_id,
                        run_id,
                        external_call_id,
                        provider,
                        model,
                        mode,
                        status,
                        latency_ms,
                        error_text,
                        metadata_json,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (external_call_id)
                    WHERE external_call_id IS NOT NULL
                    DO UPDATE SET
                        run_id = excluded.run_id,
                        provider = excluded.provider,
                        model = excluded.model,
                        mode = excluded.mode,
                        status = excluded.status,
                        latency_ms = excluded.latency_ms,
                        error_text = excluded.error_text,
                        metadata_json = excluded.metadata_json
                    RETURNING call_id
                    """,
                    [
                        resolved_call_id,
                        run_id,
                        external_call_id,
                        request.provider,
                        request.model,
                        resolved_mode,
                        resolved_status,
                        latency_ms,
                        error_text,
                        json.dumps(metadata or {}, ensure_ascii=True),
                    ],
                ).fetchone()
                persisted_call_id = str(row[0]) if row is not None else resolved_call_id

                conn.execute(
                    """
                    INSERT INTO llm_call_requests (call_id, request_json, request_hash, created_at)
                    VALUES (%s, %s::jsonb, %s, now())
                    ON CONFLICT (call_id)
                    DO UPDATE SET request_json = excluded.request_json,
                                  request_hash = excluded.request_hash
                    """,
                    [
                        persisted_call_id,
                        json.dumps(request_payload, ensure_ascii=True),
                        request_hash,
                    ],
                )

                if response_payload is not None and response is not None:
                    response_obj = response
                    conn.execute(
                        """
                        INSERT INTO llm_call_responses (
                            call_id,
                            response_json,
                            response_hash,
                            output_text,
                            finish_reason,
                            prompt_tokens,
                            completion_tokens,
                            reasoning_tokens,
                            total_tokens,
                            reasoning_text,
                            reasoning_details_json,
                            cost_total_usd,
                            cost_prompt_usd,
                            cost_completion_usd,
                            cost_reasoning_usd,
                            cost_currency,
                            cost_json,
                            created_at
                        ) VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb, now())
                        ON CONFLICT (call_id)
                        DO UPDATE SET response_json = excluded.response_json,
                                      response_hash = excluded.response_hash,
                                      output_text = excluded.output_text,
                                      finish_reason = excluded.finish_reason,
                                      prompt_tokens = excluded.prompt_tokens,
                                      completion_tokens = excluded.completion_tokens,
                                      reasoning_tokens = excluded.reasoning_tokens,
                                      total_tokens = excluded.total_tokens,
                                      reasoning_text = excluded.reasoning_text,
                                      reasoning_details_json = excluded.reasoning_details_json,
                                      cost_total_usd = excluded.cost_total_usd,
                                      cost_prompt_usd = excluded.cost_prompt_usd,
                                      cost_completion_usd = excluded.cost_completion_usd,
                                      cost_reasoning_usd = excluded.cost_reasoning_usd,
                                      cost_currency = excluded.cost_currency,
                                      cost_json = excluded.cost_json
                        """,
                        [
                            persisted_call_id,
                            json.dumps(response_payload, ensure_ascii=True),
                            response_hash,
                            response_obj.text,
                            response_obj.finish_reason,
                            response_obj.usage.prompt_tokens,
                            response_obj.usage.completion_tokens,
                            response_obj.usage.reasoning_tokens,
                            response_obj.usage.total_tokens,
                            response_obj.reasoning,
                            json.dumps(
                                response_obj.reasoning_details, ensure_ascii=True
                            )
                            if response_obj.reasoning_details is not None
                            else None,
                            response_obj.cost.total_cost_usd
                            if response_obj.cost is not None
                            else None,
                            response_obj.cost.prompt_cost_usd
                            if response_obj.cost is not None
                            else None,
                            response_obj.cost.completion_cost_usd
                            if response_obj.cost is not None
                            else None,
                            response_obj.cost.reasoning_cost_usd
                            if response_obj.cost is not None
                            else None,
                            response_obj.cost.currency
                            if response_obj.cost is not None
                            else None,
                            json.dumps(response_obj.cost.raw, ensure_ascii=True)
                            if response_obj.cost is not None
                            else None,
                        ],
                    )

                conn.commit()
                return persisted_call_id
            except errors.UniqueViolation as exc:
                conn.rollback()
                raise TransientPersistenceError(
                    f"Unique constraint conflict while recording call: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to record call: {exc}") from exc

    def record_calls_batch(
        self,
        *,
        requests: list[LlmRequest],
        responses: list[LlmResponse | None],
        run_id: str | None = None,
    ) -> list[str]:
        if len(requests) != len(responses):
            raise ValueError("requests and responses must have same length")
        call_ids: list[str] = []
        for request, response in zip(requests, responses, strict=True):
            call_ids.append(
                self.record_call(request=request, response=response, run_id=run_id)
            )
        return call_ids

    def list_calls(
        self, *, run_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[RecordedCall]:
        self.init_schema()
        with self._conn() as conn:
            if run_id is not None:
                rows = conn.execute(
                    """
                    SELECT
                        lc.call_id,
                        lc.run_id,
                        lc.provider,
                        lc.model,
                        lc.mode,
                        lc.status,
                        lc.created_at,
                        lc.latency_ms,
                        lc.error_text,
                        lcs.reasoning_tokens,
                        lcs.reasoning_text,
                        lcs.cost_total_usd,
                        lcs.cost_prompt_usd,
                        lcs.cost_completion_usd,
                        lcs.cost_reasoning_usd,
                        lcr.request_json,
                        lcs.response_json
                    FROM llm_calls lc
                    LEFT JOIN llm_call_requests lcr ON lcr.call_id = lc.call_id
                    LEFT JOIN llm_call_responses lcs ON lcs.call_id = lc.call_id
                    WHERE lc.run_id = %s
                    ORDER BY lc.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    [run_id, int(limit), int(offset)],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        lc.call_id,
                        lc.run_id,
                    lc.provider,
                    lc.model,
                    lc.mode,
                    lc.status,
                    lc.created_at,
                    lc.latency_ms,
                    lc.error_text,
                    lcs.reasoning_tokens,
                    lcs.reasoning_text,
                    lcs.cost_total_usd,
                    lcs.cost_prompt_usd,
                    lcs.cost_completion_usd,
                    lcs.cost_reasoning_usd,
                    lcr.request_json,
                    lcs.response_json
                    FROM llm_calls lc
                    LEFT JOIN llm_call_requests lcr ON lcr.call_id = lc.call_id
                    LEFT JOIN llm_call_responses lcs ON lcs.call_id = lc.call_id
                    ORDER BY lc.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    [int(limit), int(offset)],
                ).fetchall()

        out: list[RecordedCall] = []
        for row in rows:
            out.append(
                RecordedCall(
                    call_id=str(row[0]),
                    run_id=str(row[1]) if row[1] is not None else None,
                    provider=str(row[2]),
                    model=str(row[3]),
                    mode=row[4],
                    status=str(row[5]),
                    created_at=row[6],
                    latency_ms=int(row[7]),
                    error_text=str(row[8]) if row[8] is not None else None,
                    reasoning_tokens=int(row[9] or 0),
                    reasoning_text=str(row[10]) if row[10] is not None else None,
                    cost_total_usd=float(row[11]) if row[11] is not None else None,
                    cost_prompt_usd=float(row[12]) if row[12] is not None else None,
                    cost_completion_usd=float(row[13]) if row[13] is not None else None,
                    cost_reasoning_usd=float(row[14]) if row[14] is not None else None,
                    request=row[15] if isinstance(row[15], dict) else {},
                    response=row[16] if isinstance(row[16], dict) else None,
                )
            )
        return out

    def record_artifact(
        self,
        *,
        run_id: str | None,
        artifact_type: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.init_schema()
        artifact_id = uuid4().hex
        with self._conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO artifacts (artifact_id, run_id, artifact_type, artifact_path, metadata_json, created_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, now())
                    """,
                    [
                        artifact_id,
                        run_id,
                        artifact_type,
                        artifact_path,
                        json.dumps(metadata or {}, ensure_ascii=True),
                    ],
                )
                conn.commit()
                return artifact_id
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to record artifact: {exc}") from exc

    def start_session(
        self,
        *,
        strategy_mode: ToolPolicy = ToolPolicy.native_preferred,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionHandle:
        self.init_schema()
        sid = session_id or uuid4().hex
        now = utcnow()
        with self._conn() as conn:
            try:
                conn.execute(
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
                )
                conn.commit()
                return SessionHandle(
                    session_id=sid,
                    status=SessionStatus.active,
                    version=1,
                    strategy_mode=strategy_mode,
                )
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to start session: {exc}") from exc

    def get_session(self, *, session_id: str) -> SessionState:
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        turn_id = uuid4().hex
        with self._conn() as conn:
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
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        eid = event_id or uuid4().hex
        with self._conn() as conn:
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
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        tcid = tool_call_id or uuid4().hex
        with self._conn() as conn:
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
        self.init_schema()
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be >= 1")

        with self._conn() as conn:
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
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        with self._conn() as conn:
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
        self.init_schema()
        status = ToolCallStatus.succeeded if result.ok else ToolCallStatus.failed
        with self._conn() as conn:
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
                        json.dumps(result.error, ensure_ascii=True)
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
                        json.dumps(result.error, ensure_ascii=True)
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
        self.init_schema()
        dead_id = uuid4().hex
        with self._conn() as conn:
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
