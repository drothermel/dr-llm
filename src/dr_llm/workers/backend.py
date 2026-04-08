from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel


class ErrorDecision(StrEnum):
    retry = "retry"
    fail = "fail"


type ProcessFn[TWorkItem, TResult] = Callable[[TWorkItem], TResult]


class WorkerBackend[TWorkItem, TResult, TBackendState: BaseModel](Protocol):
    def claim(self, *, worker_id: str, lease_seconds: int) -> TWorkItem | None: ...

    def complete(self, *, item: TWorkItem, result: TResult, worker_id: str) -> None: ...

    def handle_process_error(
        self,
        *,
        item: TWorkItem,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision: ...

    def snapshot(self) -> TBackendState | None: ...

    def process_context(
        self,
        *,
        item: TWorkItem,
        worker_id: str,
    ) -> AbstractContextManager[Any]: ...
