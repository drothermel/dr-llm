from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from psycopg import errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_llm.errors import PersistenceError, TransientPersistenceError
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode
from dr_llm.storage._runtime import (
    StorageRuntime,
    hash_payload,
    is_retryable_db_error,
)
from dr_llm.storage.models import RecordedCall, RunStatus


class RunsCallsStore:
    def __init__(self, runtime: StorageRuntime) -> None:
        self._runtime = runtime

    def _record_call_with_conn(
        self,
        *,
        conn: Any,
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
        resolved_call_id = call_id or uuid4().hex
        request_payload = request.model_dump(mode="json", exclude_computed_fields=True)
        response_payload = (
            response.model_dump(mode="json", exclude_computed_fields=True)
            if response is not None
            else None
        )
        request_hash = hash_payload(request_payload)
        response_hash = (
            hash_payload(response_payload) if response_payload is not None else None
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
        latency_ms = int(response.latency_ms) if response is not None else None

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
                    warnings_json,
                    created_at
                ) VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
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
                              cost_json = excluded.cost_json,
                              warnings_json = excluded.warnings_json
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
                    json.dumps(response_obj.reasoning_details, ensure_ascii=True)
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
                    json.dumps(
                        [
                            warning.model_dump(
                                mode="json",
                                exclude_none=True,
                                exclude_computed_fields=True,
                            )
                            for warning in response_obj.warnings
                        ],
                        ensure_ascii=True,
                    ),
                ],
            )

        return persisted_call_id

    @retry(
        retry=retry_if_exception(is_retryable_db_error),
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
        self._runtime.init_schema()
        resolved_run_id = run_id or uuid4().hex
        with self._runtime.conn() as conn:
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
        self._runtime.init_schema()
        if not parameters:
            return 0
        with self._runtime.conn() as conn:
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
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
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
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            try:
                persisted_call_id = self._record_call_with_conn(
                    conn=conn,
                    request=request,
                    response=response,
                    run_id=run_id,
                    status=status,
                    mode=mode,
                    error_text=error_text,
                    external_call_id=external_call_id,
                    metadata=metadata,
                    call_id=call_id,
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
        self._runtime.init_schema()
        call_ids: list[str] = []
        with self._runtime.conn() as conn:
            try:
                for request, response in zip(requests, responses, strict=True):
                    call_ids.append(
                        self._record_call_with_conn(
                            conn=conn,
                            request=request,
                            response=response,
                            run_id=run_id,
                        )
                    )
                conn.commit()
                return call_ids
            except errors.UniqueViolation as exc:
                conn.rollback()
                raise TransientPersistenceError(
                    f"Unique constraint conflict while recording call batch: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(f"Failed to record call batch: {exc}") from exc

    def list_calls(
        self, *, run_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[RecordedCall]:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
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
                    lcs.warnings_json,
                    lcr.request_json,
                    lcs.response_json
                FROM llm_calls lc
                LEFT JOIN llm_call_requests lcr ON lcr.call_id = lc.call_id
                LEFT JOIN llm_call_responses lcs ON lcs.call_id = lc.call_id
                WHERE (%s::text IS NULL OR lc.run_id = %s)
                ORDER BY lc.created_at DESC
                LIMIT %s OFFSET %s
                """,
                [run_id, run_id, int(limit), int(offset)],
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
                    status=RunStatus(str(row[5])),
                    created_at=row[6],
                    latency_ms=int(row[7]) if row[7] is not None else None,
                    error_text=str(row[8]) if row[8] is not None else None,
                    reasoning_tokens=int(row[9] or 0),
                    reasoning_text=str(row[10]) if row[10] is not None else None,
                    cost_total_usd=float(row[11]) if row[11] is not None else None,
                    cost_prompt_usd=float(row[12]) if row[12] is not None else None,
                    cost_completion_usd=float(row[13]) if row[13] is not None else None,
                    cost_reasoning_usd=float(row[14]) if row[14] is not None else None,
                    warnings=(row[15] if isinstance(row[15], list) else []),
                    request=row[16] if isinstance(row[16], dict) else {},
                    response=row[17] if isinstance(row[17], dict) else None,
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
        self._runtime.init_schema()
        artifact_id = uuid4().hex
        with self._runtime.conn() as conn:
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
