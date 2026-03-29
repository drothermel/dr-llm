from __future__ import annotations

from typing import Any

from dr_llm.pool.call_recorder import CallRecorder
from dr_llm.pool.recorded_call import RecordedCall, RunStatus
from dr_llm.pool.runtime import DbConfig, DbRuntime
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode


def try_init_db_from_dsn(dsn: str | None = None) -> None:
    db = PoolDb(dsn=dsn)
    try:
        db.initialize()
    finally:
        db.close()


class PoolDb:
    def __init__(
        self,
        config: DbConfig | None = None,
        dsn: str | None = None,
    ) -> None:
        self.config = config
        if self.config is None:
            self.config = DbConfig() if dsn is None else DbConfig(dsn=dsn)

        self._runtime = DbRuntime(self.config)
        self._recorder = CallRecorder(self._runtime)

    @property
    def runtime(self) -> DbRuntime:
        return self._runtime

    def close(self) -> None:
        self._runtime.close()

    def init_schema(self) -> None:
        self._runtime.init_schema()

    def initialize(self) -> None:
        self._runtime.initialize()

    def start_run(
        self,
        *,
        run_type: str = "generic",
        status: RunStatus = RunStatus.running,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        return self._recorder.start_run(
            run_type=run_type,
            status=status,
            metadata=metadata,
            run_id=run_id,
        )

    def upsert_run_parameters(self, *, run_id: str, parameters: dict[str, Any]) -> int:
        return self._recorder.upsert_run_parameters(
            run_id=run_id,
            parameters=parameters,
        )

    def finish_run(
        self,
        *,
        run_id: str,
        status: RunStatus,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._recorder.finish_run(run_id=run_id, status=status, metadata=metadata)

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
        return self._recorder.record_call(
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

    def record_calls_batch(
        self,
        *,
        requests: list[LlmRequest],
        responses: list[LlmResponse | None],
        run_id: str | None = None,
    ) -> list[str]:
        return self._recorder.record_calls_batch(
            requests=requests,
            responses=responses,
            run_id=run_id,
        )

    def list_calls(
        self,
        *,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RecordedCall]:
        return self._recorder.list_calls(run_id=run_id, limit=limit, offset=offset)

    def record_artifact(
        self,
        *,
        run_id: str | None,
        artifact_type: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._recorder.record_artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            metadata=metadata,
        )
